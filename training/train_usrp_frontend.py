#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
USRP 预对接前端训练脚本。

目标：
- 在没有 USRP 的情况下，先把“同步 / 信道估计 / 均衡”中的可训练残差学起来
- 后续拿真实录制 IQ 直接回放微调

训练对象：
- ResidualChannelEstimator
- ResidualEqualizer
- PilotSynchronizer / LS / MMSE 保持传统基线，不训练

训练数据来源：
1) synthetic：脚本内合成的平坦衰落 + CFO + AWGN
2) npz：实验室录制回放数据

输出：
- checkpoints/usrp_prep_frontend/ckpt-*
- results/usrp_prep_frontend_train.json
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from edge_semcom.usrp_frontend import USRPReceiverFrontEnd, synthetic_burst
from evaluate_usrp_prep_frontend import load_recording_npz, metrics_from_xhat

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
tf.get_logger().setLevel("ERROR")


def complex_mse(x_true: tf.Tensor, x_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.abs(x_true - x_pred) ** 2)


def qpsk_demod_hard(x):
    b0 = tf.cast(tf.math.real(x) > 0.0, tf.float32)
    b1 = tf.cast(tf.math.imag(x) > 0.0, tf.float32)
    return tf.stack([b0, b1], axis=-1)


def symbol_metrics_from_xhat(x_true: tf.Tensor, x_hat: tf.Tensor):
    bits_true = qpsk_demod_hard(x_true)
    bits_hat = qpsk_demod_hard(x_hat)
    ber = tf.reduce_mean(tf.abs(bits_true - bits_hat))
    evm = tf.reduce_mean(tf.abs(x_hat - x_true) ** 2 / (tf.abs(x_true) ** 2 + 1e-8))
    return float(ber.numpy()), float(evm.numpy())


def make_batch_from_recording(rec, batch_size: int, pilot_len: int, data_len: int):
    bits = rec["bits"]
    x_data = rec["x_data"]
    x_pilot = rec["x_pilot"]
    y_pilot = rec["y_pilot"]
    y_data = rec["y_data"]
    no = rec["no"]

    n = int(bits.shape[0])
    if n == 0:
        raise ValueError("录制文件为空")

    idx = np.random.randint(0, n, size=batch_size)
    return (
        tf.gather(bits, idx),
        tf.gather(x_data, idx),
        tf.gather(x_pilot, idx),
        tf.gather(y_pilot, idx),
        tf.gather(y_data, idx),
        tf.gather(no, idx),
    )


def make_batch_synth(batch_size: int, pilot_len: int, data_len: int, snr_db: float):
    return synthetic_burst(
        batch_size=batch_size,
        num_pilot=pilot_len,
        num_data=data_len,
        snr_db=snr_db,
    )


def train_one_mode(args, mode: str, recording=None):
    if mode == "pilot_sync":
        frontend = USRPReceiverFrontEnd(
            use_residual_ce=False,
            use_residual_eq=False,
            estimator_type="lmmse_dft",
            equalizer_type="iterative",
            equalizer_iters=2,
            dft_taps=4,
        )
    elif mode == "residual":
        frontend = USRPReceiverFrontEnd(
            use_residual_ce=True,
            use_residual_eq=True,
            estimator_type="lmmse_dft",
            equalizer_type="iterative",
            equalizer_iters=2,
            dft_taps=4,
        )
    else:
        raise ValueError(mode)

    # build once so trainable_variables is populated for residual mode
    if recording is None:
        x_pilot0, x_data0, y_pilot0, y_data0, no0 = make_batch_synth(
            args.batch_size, args.pilot_len, args.data_len, args.snr_db
        )
    else:
        _, x_data0, x_pilot0, y_pilot0, y_data0, no0 = make_batch_from_recording(
            recording, args.batch_size, args.pilot_len, args.data_len
        )
    _ = frontend(y_pilot0, x_pilot0, y_data0, no=no0)

    # 只有 residual 模式真的需要训练
    trainable_vars = frontend.trainable_variables
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    ckpt_dir = os.path.join(args.checkpoint_dir, mode)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = tf.train.Checkpoint(frontend=frontend, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)

    history = {
        "loss": [],
        "ber": [],
        "evm": [],
    }

    best_loss = float("inf")
    best_path = None

    @tf.function(reduce_retracing=True)
    def train_step(x_pilot, x_data, y_pilot, y_data, no):
        with tf.GradientTape() as tape:
            out = frontend(y_pilot, x_pilot, y_data, no=no)
            x_hat = out["x_hat"]
            loss = complex_mse(x_data, x_hat)
        if trainable_vars:
            grads = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
        return loss, x_hat

    print("=" * 80)
    print(f"[TRAIN] mode={mode}")
    print(f"batch_size={args.batch_size}, pilot_len={args.pilot_len}, data_len={args.data_len}, steps={args.steps}")
    print(f"data_source={'npz' if recording is not None else 'synthetic'}, lr={args.lr}")
    print(f"trainable vars: {len(trainable_vars)}")
    print("=" * 80)

    for step in range(1, args.steps + 1):
        if recording is None:
            x_pilot, x_data, y_pilot, y_data, no = make_batch_synth(
                args.batch_size, args.pilot_len, args.data_len, args.snr_db
            )
        else:
            _, x_data, x_pilot, y_pilot, y_data, no = make_batch_from_recording(
                recording, args.batch_size, args.pilot_len, args.data_len
            )

        loss, x_hat = train_step(x_pilot, x_data, y_pilot, y_data, no)

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            ber, evm = symbol_metrics_from_xhat(x_data, x_hat)
            loss_v = float(loss.numpy())
            history["loss"].append(loss_v)
            history["ber"].append(ber)
            history["evm"].append(evm)
            print(f"step {step:5d}/{args.steps} | loss={loss_v:.6f} | ber={ber:.6f} | evm={evm:.6f}")

            if loss_v < best_loss:
                best_loss = loss_v
                best_path = ckpt_manager.save()
                print(f"  ★ saved best checkpoint: {best_path}")

    final_path = ckpt_manager.save()
    print(f"[DONE] final checkpoint: {final_path}")

    return {
        "mode": mode,
        "best_loss": best_loss,
        "best_path": best_path,
        "final_path": final_path,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", choices=["synthetic", "npz"], default="synthetic")
    parser.add_argument("--input_npz", type=str, default="", help="录制回放 NPZ（data_source=npz 时使用）")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/usrp_prep_frontend")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pilot_len", type=int, default=16)
    parser.add_argument("--data_len", type=int, default=256)
    parser.add_argument("--snr_db", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    if args.data_source == "npz":
        if not args.input_npz:
            raise ValueError("data_source=npz 时必须提供 --input_npz")
        recording = load_recording_npz(args.input_npz)
    else:
        recording = None

    os.makedirs("results", exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    results = []

    # 先跑传统基线（不训练）
    base = train_one_mode(args, "pilot_sync", recording=recording)
    results.append(base)

    # residual 模式可训练；eval_only 时只做冒烟/基线
    if not args.eval_only:
        res = train_one_mode(args, "residual", recording=recording)
        results.append(res)

    out_path = os.path.join("results", "usrp_prep_frontend_train.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for item in results:
        mode = item["mode"]
        h = item["history"]
        axes[0].plot(h["loss"], label=mode)
        axes[1].plot(h["ber"], label=mode)
        axes[2].plot(h["evm"], label=mode)
    axes[0].set_title("Train Loss")
    axes[1].set_title("Train BER")
    axes[2].set_title("Train EVM")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    png_path = os.path.join("results", "usrp_prep_frontend_train.png")
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()

