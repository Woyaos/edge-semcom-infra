#!/usr/bin/env python3
"""
工业化增量训练脚本（稳定版）

目标：不要一步到位替换整条物理层，而是先训练“可收敛”的分段神经物理层。

核心思路：
1) 把 7500 bit 拆成多个 segment（默认每段 750 bit，共 10 段）
2) 仅训练单段模型（750 -> 250 complex symbols），更容易收敛
3) 后续在 TX/RX 里按段循环调用该模型完成全量替换
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from edge_semcom.neural_phy import NeuralPhysicalLayer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_bits(batch_size: int, k: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, 2, (batch_size, k), dtype=torch.float32, device=device)


def sample_snr(
    batch_size: int,
    snr_min: float,
    snr_max: float,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(batch_size, device=device).uniform_(snr_min, snr_max)


def compute_ber(logits: torch.Tensor, target_bits: torch.Tensor) -> float:
    hard = (logits > 0).float()
    ber = (hard != target_bits).float().mean().item()
    return float(ber)


class Trainer:
    def __init__(self, model: NeuralPhysicalLayer, device: torch.device, lr: float, weight_decay: float):
        self.model = model
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def train_epoch(
        self,
        steps_per_epoch: int,
        batch_size: int,
        k: int,
        train_snr_min: float,
        train_snr_max: float,
    ) -> Dict[str, float]:
        self.model.train()
        losses = []
        bers = []

        for _ in range(steps_per_epoch):
            bits = sample_bits(batch_size, k, self.device)
            snr_db = sample_snr(batch_size, train_snr_min, train_snr_max, self.device)

            out = self.model(bits, snr_db, channel_type="awgn")
            logits = out["bit_logits"]

            loss = self.criterion(logits, bits)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            losses.append(loss.item())
            bers.append(compute_ber(logits.detach(), bits))

        self.scheduler.step()

        return {
            "loss": float(np.mean(losses)),
            "ber": float(np.mean(bers)),
        }

    @torch.no_grad()
    def validate(
        self,
        val_steps: int,
        batch_size: int,
        k: int,
        eval_snr_db: float,
    ) -> Dict[str, float]:
        self.model.eval()
        losses = []
        bers = []

        for _ in range(val_steps):
            bits = sample_bits(batch_size, k, self.device)
            snr_db = torch.full((batch_size,), eval_snr_db, dtype=torch.float32, device=self.device)

            out = self.model(bits, snr_db, channel_type="awgn")
            logits = out["bit_logits"]

            loss = self.criterion(logits, bits)
            losses.append(loss.item())
            bers.append(compute_ber(logits, bits))

        return {
            "loss": float(np.mean(losses)),
            "ber": float(np.mean(bers)),
        }


def curriculum_snr(epoch: int, snr_min: float, snr_max: float) -> Tuple[float, float]:
    # 工业实践：先高SNR，后扩展低SNR
    if epoch <= 5:
        return max(8.0, snr_min), snr_max
    if epoch <= 15:
        return max(5.0, snr_min), snr_max
    if epoch <= 30:
        return max(2.0, snr_min), snr_max
    return snr_min, snr_max


def save_checkpoint(path: Path, model: NeuralPhysicalLayer, config: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "config": config,
    }
    torch.save(ckpt, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="工业化增量训练：分段神经物理层")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps_per_epoch", type=int, default=300)
    parser.add_argument("--val_steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--snr_min", type=float, default=0.0)
    parser.add_argument("--snr_max", type=float, default=15.0)
    parser.add_argument("--eval_snr", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="models")

    # 分段参数（关键）
    parser.add_argument("--total_k", type=int, default=7500, help="总比特数")
    parser.add_argument("--segments", type=int, default=10, help="分段数")
    parser.add_argument("--segment_n_symbols", type=int, default=250, help="每段符号数")
    parser.add_argument("--encoder_hidden", type=int, default=256)
    parser.add_argument("--decoder_hidden", type=int, default=256)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.total_k % args.segments != 0:
        raise ValueError(f"total_k={args.total_k} 必须能被 segments={args.segments} 整除")

    segment_k = args.total_k // args.segments
    segment_n = args.segment_n_symbols

    print("=" * 72)
    print("工业化增量训练（分段）")
    print(f"设备: {device}")
    print(f"总比特: {args.total_k}, 分段: {args.segments}, 每段比特: {segment_k}")
    print(f"每段符号: {segment_n}")
    print("=" * 72)

    model = NeuralPhysicalLayer(
        k=segment_k,
        n_symbols=segment_n,
        encoder_hidden=args.encoder_hidden,
        decoder_hidden=args.decoder_hidden,
    ).to(device)

    trainer = Trainer(model, device, lr=args.lr, weight_decay=args.weight_decay)

    history = {
        "train_loss": [],
        "train_ber": [],
        "val_loss": [],
        "val_ber": [],
    }

    best_ber = float("inf")
    save_dir = Path(args.save_dir)
    best_path = save_dir / "neural_phy_segment_best.pt"
    last_path = save_dir / "neural_phy_segment_last.pt"
    hist_path = save_dir / "training_history_segment.json"

    config = {
        "total_k": args.total_k,
        "segments": args.segments,
        "segment_k": segment_k,
        "segment_n_symbols": segment_n,
        "encoder_hidden": args.encoder_hidden,
        "decoder_hidden": args.decoder_hidden,
    }

    for epoch in range(1, args.epochs + 1):
        train_snr_min, train_snr_max = curriculum_snr(epoch, args.snr_min, args.snr_max)
        train_m = trainer.train_epoch(
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            k=segment_k,
            train_snr_min=train_snr_min,
            train_snr_max=train_snr_max,
        )
        val_m = trainer.validate(
            val_steps=args.val_steps,
            batch_size=args.batch_size,
            k=segment_k,
            eval_snr_db=args.eval_snr,
        )

        history["train_loss"].append(train_m["loss"])
        history["train_ber"].append(train_m["ber"])
        history["val_loss"].append(val_m["loss"])
        history["val_ber"].append(val_m["ber"])

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train SNR[{train_snr_min:.1f},{train_snr_max:.1f}] | "
            f"Train Loss={train_m['loss']:.5f} BER={train_m['ber']:.5f} | "
            f"Val Loss={val_m['loss']:.5f} BER={val_m['ber']:.5f}"
        )

        save_checkpoint(last_path, model, config)
        if val_m["ber"] < best_ber:
            best_ber = val_m["ber"]
            save_checkpoint(best_path, model, config)
            print(f"  ✓ 新最佳模型: {best_path} (BER={best_ber:.6f})")

        save_dir.mkdir(parents=True, exist_ok=True)
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("训练完成")
    print(f"最佳模型: {best_path}")
    print(f"最后模型: {last_path}")
    print(f"历史文件: {hist_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

