#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在无 USRP 条件下评估“算法侧”前端（同步/信道估计/均衡）的性能。

输出：
- 控制台表格（BER / EVM / latency）
- results/usrp_prep_frontend_eval.json

说明：
- residual 模式默认是“未训练残差网络”，仅用于检查可运行性，不代表最终性能。
- 建议论文对比重点看：raw vs pilot_sync（传统可解释）
"""

import os
import json
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from edge_semcom.usrp_frontend import USRPReceiverFrontEnd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
tf.get_logger().setLevel("ERROR")


def qpsk_mod(bits):
    """bits: [B, N, 2] -> complex symbols [B, N]"""
    b0 = 2.0 * bits[..., 0] - 1.0
    b1 = 2.0 * bits[..., 1] - 1.0
    s = tf.complex(b0, b1) / tf.cast(tf.sqrt(tf.constant(2.0, dtype=tf.float32)), tf.complex64)
    return s


def qpsk_demod_hard(x):
    """x: [B,N] complex -> hard bits [B,N,2] float32"""
    b0 = tf.cast(tf.math.real(x) > 0.0, tf.float32)
    b1 = tf.cast(tf.math.imag(x) > 0.0, tf.float32)
    return tf.stack([b0, b1], axis=-1)


def build_pilot(batch_size, pilot_len):
    p = tf.complex(
        tf.ones([batch_size, pilot_len], dtype=tf.float32),
        tf.ones([batch_size, pilot_len], dtype=tf.float32),
    )
    return p / tf.cast(tf.sqrt(tf.constant(2.0, dtype=tf.float32)), tf.complex64)


def make_burst(batch_size=64, pilot_len=16, data_len=256, snr_db=8.0, cfo_strength=0.03):
    # payload bits/symbols
    bits = tf.cast(tf.random.uniform([batch_size, data_len, 2], minval=0, maxval=2, dtype=tf.int32), tf.float32)
    x_data = qpsk_mod(bits)

    # pilot
    x_pilot = build_pilot(batch_size, pilot_len)

    # flat-fading channel
    h = tf.complex(
        tf.random.normal([batch_size, 1], stddev=0.7),
        tf.random.normal([batch_size, 1], stddev=0.7),
    )

    # per-symbol phase drift (mimic CFO)
    n = tf.cast(tf.range(pilot_len + data_len)[None, :], tf.float32)
    w = tf.random.uniform([batch_size, 1], minval=-cfo_strength, maxval=cfo_strength)
    phase = w * n
    rot = tf.exp(tf.complex(tf.zeros_like(phase), phase))

    x_all = tf.concat([x_pilot, x_data], axis=1)
    y_all = h * x_all * rot

    # noise
    no = tf.ones([batch_size, 1], tf.float32) * tf.pow(10.0, -snr_db / 10.0)
    noise = tf.complex(
        tf.random.normal(tf.shape(y_all), stddev=tf.sqrt(no / 2.0)),
        tf.random.normal(tf.shape(y_all), stddev=tf.sqrt(no / 2.0)),
    )
    y_all = y_all + noise

    y_pilot = y_all[:, :pilot_len]
    y_data = y_all[:, pilot_len:]
    return bits, x_data, x_pilot, y_pilot, y_data, no


def metrics_from_xhat(bits_true, x_true, x_hat):
    bits_hat = qpsk_demod_hard(x_hat)
    ber = tf.reduce_mean(tf.abs(bits_true - bits_hat))
    evm = tf.reduce_mean(tf.abs(x_hat - x_true) ** 2 / (tf.abs(x_true) ** 2 + 1e-8))
    return float(ber.numpy()), float(evm.numpy())


def load_recording_npz(path):
    """加载实验录制数据，要求至少包含 bits/x_data/x_pilot/y_pilot/y_data/no。"""
    obj = np.load(path, allow_pickle=False)
    required = ["bits", "x_data", "x_pilot", "y_pilot", "y_data", "no"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise KeyError(f"录制文件缺少字段: {missing}")
    return {
        "bits": tf.convert_to_tensor(obj["bits"], dtype=tf.float32),
        "x_data": tf.convert_to_tensor(obj["x_data"], dtype=tf.complex64),
        "x_pilot": tf.convert_to_tensor(obj["x_pilot"], dtype=tf.complex64),
        "y_pilot": tf.convert_to_tensor(obj["y_pilot"], dtype=tf.complex64),
        "y_data": tf.convert_to_tensor(obj["y_data"], dtype=tf.complex64),
        "no": tf.convert_to_tensor(obj["no"], dtype=tf.float32),
    }


def benchmark_frontend(mode, loops=60, warmup=10, batch_size=64, pilot_len=16, data_len=256, snr_db=8.0, recording=None):
    if mode == "pilot_sync":
        frontend = USRPReceiverFrontEnd(use_residual_ce=False, use_residual_eq=False)
    elif mode == "lmmse_dft_iter2":
        frontend = USRPReceiverFrontEnd(
            use_residual_ce=False,
            use_residual_eq=False,
            estimator_type="lmmse_dft",
            equalizer_type="iterative",
            equalizer_iters=2,
            dft_taps=4,
        )
    elif mode == "residual_untrained":
        frontend = USRPReceiverFrontEnd(
            use_residual_ce=True,
            use_residual_eq=True,
            estimator_type="lmmse_dft",
            equalizer_type="iterative",
            equalizer_iters=2,
            dft_taps=4,
        )
    else:
        frontend = None

    ber_list = []
    evm_list = []
    lat_list = []

    def get_sample():
        if recording is None:
            return make_burst(batch_size, pilot_len, data_len, snr_db)
        return (
            recording["bits"],
            recording["x_data"],
            recording["x_pilot"],
            recording["y_pilot"],
            recording["y_data"],
            recording["no"],
        )

    # warmup
    for _ in range(warmup):
        bits, x_data, x_pilot, y_pilot, y_data, no = get_sample()
        if mode == "raw":
            x_hat = y_data
        elif mode in ("pilot_sync", "lmmse_dft_iter2", "residual_untrained"):
            out = frontend(y_pilot, x_pilot, y_data, no=no)
            x_hat = out["x_hat"]
        else:
            raise ValueError(mode)
        _ = metrics_from_xhat(bits, x_data, x_hat)

    # measure
    for _ in range(loops):
        bits, x_data, x_pilot, y_pilot, y_data, no = get_sample()

        t0 = time.time()
        if mode == "raw":
            x_hat = y_data
        elif mode in ("pilot_sync", "lmmse_dft_iter2", "residual_untrained"):
            out = frontend(y_pilot, x_pilot, y_data, no=no)
            x_hat = out["x_hat"]
        else:
            raise ValueError(mode)
        t1 = time.time()

        ber, evm = metrics_from_xhat(bits, x_data, x_hat)
        ber_list.append(ber)
        evm_list.append(evm)
        lat_list.append((t1 - t0) * 1000.0)

    return {
        "ber": float(np.mean(ber_list)),
        "evm": float(np.mean(evm_list)),
        "latency_ms": float(np.mean(lat_list)),
        "latency_p95_ms": float(np.percentile(lat_list, 95)),
    }


def benchmark_with_frontend(frontend, loops=60, warmup=10, batch_size=64, pilot_len=16, data_len=256, snr_db=8.0, recording=None):
    """对外部传入（可加载训练权重）的前端做同口径评估。"""
    ber_list = []
    evm_list = []
    lat_list = []

    def get_sample():
        if recording is None:
            return make_burst(batch_size, pilot_len, data_len, snr_db)
        return (
            recording["bits"],
            recording["x_data"],
            recording["x_pilot"],
            recording["y_pilot"],
            recording["y_data"],
            recording["no"],
        )

    for _ in range(warmup):
        bits, x_data, x_pilot, y_pilot, y_data, no = get_sample()
        out = frontend(y_pilot, x_pilot, y_data, no=no)
        _ = metrics_from_xhat(bits, x_data, out["x_hat"])

    for _ in range(loops):
        bits, x_data, x_pilot, y_pilot, y_data, no = get_sample()
        t0 = time.time()
        out = frontend(y_pilot, x_pilot, y_data, no=no)
        t1 = time.time()

        ber, evm = metrics_from_xhat(bits, x_data, out["x_hat"])
        ber_list.append(ber)
        evm_list.append(evm)
        lat_list.append((t1 - t0) * 1000.0)

    return {
        "ber": float(np.mean(ber_list)),
        "evm": float(np.mean(evm_list)),
        "latency_ms": float(np.mean(lat_list)),
        "latency_p95_ms": float(np.percentile(lat_list, 95)),
    }


def plot_eval_results(results: dict, out_png: str):
    modes = list(results.keys())
    ber = [results[m]["ber"] for m in modes]
    evm = [results[m]["evm"] for m in modes]
    lat = [results[m]["latency_ms"] for m in modes]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(modes, ber)
    axes[0].set_title("BER (lower is better)")
    axes[0].set_ylabel("BER")
    axes[0].tick_params(axis='x', rotation=20)

    axes[1].bar(modes, evm)
    axes[1].set_title("EVM (lower is better)")
    axes[1].set_ylabel("EVM")
    axes[1].tick_params(axis='x', rotation=20)

    axes[2].bar(modes, lat)
    axes[2].set_title("Latency ms (lower is better)")
    axes[2].set_ylabel("ms")
    axes[2].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def run_pilot_scan(pilot_list, mode, loops, warmup, batch_size, data_len, snr_db, recording=None):
    rows = []
    for p in pilot_list:
        r = benchmark_frontend(
            mode,
            loops=loops,
            warmup=warmup,
            batch_size=batch_size,
            pilot_len=int(p),
            data_len=data_len,
            snr_db=snr_db,
            recording=recording,
        )
        rows.append({"pilot_len": int(p), **r})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", type=str, default="", help="实验录制数据 NPZ，包含 bits/x_data/x_pilot/y_pilot/y_data/no")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pilot_len", type=int, default=16)
    parser.add_argument("--data_len", type=int, default=256)
    parser.add_argument("--snr_db", type=float, default=8.0)
    parser.add_argument("--loops", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--residual_ckpt_dir", type=str, default="", help="已训练 residual 前端 checkpoint 目录")
    parser.add_argument("--residual_ckpt_path", type=str, default="", help="已训练 residual 前端 checkpoint 文件路径（优先于目录）")
    parser.add_argument("--pilot_scan", type=str, default="", help="导频扫描，例如 8,16,24,32")
    parser.add_argument("--pilot_scan_mode", type=str, default="lmmse_dft_iter2", choices=["pilot_sync", "lmmse_dft_iter2", "residual_untrained"], help="导频扫描使用的模式")
    args = parser.parse_args()

    batch_size = int(args.batch_size)
    pilot_len = int(args.pilot_len)
    data_len = int(args.data_len)
    snr_db = float(args.snr_db)

    recording = load_recording_npz(args.input_npz) if args.input_npz else None

    modes = ["raw", "pilot_sync", "lmmse_dft_iter2", "residual_untrained"]
    results = {}
    for m in modes:
        results[m] = benchmark_frontend(
            m,
            loops=int(args.loops),
            warmup=int(args.warmup),
            batch_size=batch_size,
            pilot_len=pilot_len,
            data_len=data_len,
            snr_db=snr_db,
            recording=recording,
        )

    # 可选：加载训练后的 residual 前端
    restore_path = args.residual_ckpt_path
    if (not restore_path) and args.residual_ckpt_dir:
        restore_path = tf.train.latest_checkpoint(args.residual_ckpt_dir)

    if restore_path:
        frontend = USRPReceiverFrontEnd(
            use_residual_ce=True,
            use_residual_eq=True,
            estimator_type="lmmse_dft",
            equalizer_type="iterative",
            equalizer_iters=2,
            dft_taps=4,
        )
        # build once
        bits_b, x_data_b, x_pilot_b, y_pilot_b, y_data_b, no_b = make_burst(batch_size, pilot_len, data_len, snr_db)
        _ = frontend(y_pilot_b, x_pilot_b, y_data_b, no=no_b)
        ckpt = tf.train.Checkpoint(frontend=frontend)
        ckpt.restore(restore_path).expect_partial()
        results["residual_trained"] = benchmark_with_frontend(
            frontend,
            loops=int(args.loops),
            warmup=int(args.warmup),
            batch_size=batch_size,
            pilot_len=pilot_len,
            data_len=data_len,
            snr_db=snr_db,
            recording=recording,
        )

    print("\n=== USRP-Prep Frontend Eval (no USRP) ===")
    print(f"batch={batch_size}, pilot_len={pilot_len}, data_len={data_len}, snr_db={snr_db}")
    print(f"{'mode':<22} {'BER':<12} {'EVM':<12} {'lat(ms)':<12} {'p95(ms)':<12}")
    print("-" * 74)
    for m in results.keys():
        r = results[m]
        print(f"{m:<22} {r['ber']:<12.6f} {r['evm']:<12.6f} {r['latency_ms']:<12.3f} {r['latency_p95_ms']:<12.3f}")

    os.makedirs("results", exist_ok=True)
    out = {
        "config": {
            "batch_size": batch_size,
            "pilot_len": pilot_len,
            "data_len": data_len,
            "snr_db": snr_db,
            "input_npz": args.input_npz,
            "residual_ckpt_dir": args.residual_ckpt_dir,
            "residual_ckpt_path": args.residual_ckpt_path,
        },
        "results": results,
    }
    out_path = "results/usrp_prep_frontend_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")

    out_png = "results/usrp_prep_frontend_eval.png"
    plot_eval_results(results, out_png)
    print(f"Saved: {out_png}")

    if args.pilot_scan.strip():
        pilot_list = [int(x) for x in args.pilot_scan.split(",") if x.strip()]
        scan_rows = run_pilot_scan(
            pilot_list,
            mode=args.pilot_scan_mode,
            loops=int(args.loops),
            warmup=int(args.warmup),
            batch_size=batch_size,
            data_len=data_len,
            snr_db=snr_db,
            recording=recording,
        )
        scan_out = {
            "mode": args.pilot_scan_mode,
            "rows": scan_rows,
        }
        scan_path = "results/usrp_prep_frontend_pilot_scan.json"
        with open(scan_path, "w", encoding="utf-8") as f:
            json.dump(scan_out, f, ensure_ascii=False, indent=2)
        print(f"Saved: {scan_path}")


if __name__ == "__main__":
    main()

