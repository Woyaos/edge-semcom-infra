#!/usr/bin/env python3
"""
严格的端到端吞吐/时延测试脚本：对比神经版本 vs 传统版本
测试项目:
  1. TX物理层吞吐量 (codewords/sec)
  2. RX物理层时延 (ms/codeword)
  3. 端到端BER
  4. VAE编码/解码时延
  5. 总系统时延
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import time
import argparse
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Sionna imports
from sionna.phy import Block
from sionna.phy.channel import AWGN
from sionna.phy.utils import ebnodb2no, log10, expand_to_rank
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, Constellation, BinarySource
from sionna.phy.utils import sim_ber

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')

# ==================== Simulation Configuration ====================
num_bits_per_symbol = 6
modulation_order = 2**num_bits_per_symbol
coderate = 0.5
n = 15000
num_symbols_per_codeword = n // num_bits_per_symbol
k = int(n * coderate)

# Test parameters
TEST_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
TEST_SNR_DB = 6.0
NUM_TEST_CODEWORDS = 2000  # TX/E2E 默认每项处理的码字数
NUM_RX_TEST_CODEWORDS = 512  # RX 解码较慢，单独默认更小
NUM_TRIALS = 3  # Repeat each config N times

# ==================== Import Neural Components ====================
print("[INFO] Loading neural components...")
try:
    from edge_semcom.neural_modem import NeuralModulator, NeuralDemodulator
    NEURAL_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Failed to import edge_semcom.neural_modem: {e}")
    NEURAL_AVAILABLE = False

# ==================== Physical Layer Components ====================

class TraditionalTXPhysical(tf.keras.Model):
    """Traditional QAM mapper (baseline)"""
    def __init__(self):
        super().__init__()
        self.constellation = Constellation("qam", num_bits_per_symbol, trainable=False)
        self._mapper = Mapper(constellation=self.constellation)
    
    def call(self, batch_size, c, ebno_db):
        x = self._mapper(c)
        return x


class NeuralTXPhysical(tf.keras.Model):
    """Neural modulator TX"""
    def __init__(self, ckpt_dir="neural_modem_checkpoints_v2", ckpt_path=""):
        super().__init__()
        self._modulator = NeuralModulator(num_bits_per_symbol, num_symbols_per_codeword)
        
        # Dummy build
        dummy_bits = tf.zeros([1, num_symbols_per_codeword, num_bits_per_symbol], dtype=tf.float32)
        _ = self._modulator(dummy_bits, training=False)
        
        # Load checkpoint
        ckpt = tf.train.Checkpoint(modulator=self._modulator)
        restore_path = ckpt_path
        if not restore_path and ckpt_dir:
            restore_path = tf.train.latest_checkpoint(ckpt_dir)
        if restore_path:
            ckpt.restore(restore_path).expect_partial()
            print(f"[TX][NEURAL] Loaded checkpoint: {restore_path}")
        else:
            print("[TX][NEURAL] Using random initialization (no checkpoint)")
    
    def call(self, batch_size, c, ebno_db):
        c = tf.cast(c, tf.float32)
        symbols_bits = tf.reshape(
            c[:, :num_symbols_per_codeword * num_bits_per_symbol],
            [batch_size, num_symbols_per_codeword, num_bits_per_symbol],
        )
        x = self._modulator(symbols_bits, training=False)
        return x


class TraditionalRXPhysical(tf.keras.Model):
    """Traditional APP demapper"""
    def __init__(self):
        super().__init__()
        self.constellation = Constellation("qam", num_bits_per_symbol, trainable=False)
        self._demapper = Demapper("app", constellation=self.constellation)
        self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
    
    def call(self, batch_size, y, ebno_db):
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)
        llr = self._demapper(y, no)
        llr = tf.reshape(llr, [batch_size, n])
        b_hat = self._decoder(llr)
        return b_hat


class NeuralRXPhysical(tf.keras.Model):
    """Neural demapper RX"""
    def __init__(self, ckpt_dir="neural_modem_checkpoints_v2", ckpt_path="", llr_sign=1.0):
        super().__init__()
        self._demapper = NeuralDemodulator(
            num_bits_per_symbol=num_bits_per_symbol,
            num_symbols_per_codeword=num_symbols_per_codeword,
        )
        self._llr_sign = float(llr_sign)

        # Dummy build，确保 checkpoint 变量可正确匹配恢复
        dummy_y = tf.complex(
            tf.zeros([1, num_symbols_per_codeword], dtype=tf.float32),
            tf.zeros([1, num_symbols_per_codeword], dtype=tf.float32),
        )
        dummy_no = tf.ones([1, 1], dtype=tf.float32)
        _ = self._demapper(dummy_y, dummy_no, training=False)
        
        # Load checkpoint
        # train_neural_modem.py 保存键名为 demodulator
        ckpt = tf.train.Checkpoint(demodulator=self._demapper)
        restore_path = ckpt_path
        if not restore_path and ckpt_dir:
            restore_path = tf.train.latest_checkpoint(ckpt_dir)
        if restore_path:
            ckpt.restore(restore_path).expect_partial()
            print(f"[RX][NEURAL] Loaded checkpoint: {restore_path}")
        else:
            print("[RX][NEURAL] Using random initialization (no checkpoint)")
        
        self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
    
    def call(self, batch_size, y, ebno_db):
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)
        llr = self._llr_sign * self._demapper(y, no)
        llr = tf.reshape(llr, [batch_size, n])
        b_hat = self._decoder(llr)
        return b_hat


# ==================== Benchmark Functions ====================

def benchmark_tx_throughput(tx_model, batch_sizes, num_codewords=2000, num_trials=3):
    """
    Measure TX throughput (codewords/second)
    """
    encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
    channel = AWGN()
    no = ebnodb2no(tf.constant([TEST_SNR_DB], tf.float32), num_bits_per_symbol, coderate)
    no = expand_to_rank(no, 2)
    
    results = {}
    
    for batch_size in batch_sizes:
        times = []
        
        for trial in range(num_trials):
            # Generate data
            num_batches = num_codewords // batch_size
            b = tf.cast(tf.random.uniform((batch_size, k), minval=0, maxval=2, dtype=tf.int32), tf.float32)
            c = encoder(b)
            ebno_db = tf.fill([batch_size], TEST_SNR_DB)
            
            # Warmup
            _ = tx_model(batch_size=batch_size, c=c, ebno_db=ebno_db)
            
            # Benchmark
            t_start = time.perf_counter()
            for _ in range(num_batches):
                b = tf.cast(tf.random.uniform((batch_size, k), minval=0, maxval=2, dtype=tf.int32), tf.float32)
                c = encoder(b)
                x = tx_model(batch_size=batch_size, c=c, ebno_db=ebno_db)
                y = channel(x, no)
            t_end = time.perf_counter()
            
            throughput = (batch_size * num_batches) / (t_end - t_start)  # codewords/sec
            times.append(throughput)
        
        results[batch_size] = {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "times": times,
        }
        print(f"  TX Batch {batch_size:3d}: {np.mean(times):8.1f} ± {np.std(times):6.1f} codewords/sec")
    
    return results


def benchmark_rx_latency(rx_model, batch_sizes, num_codewords=512, num_trials=3, progress=True):
    """
    Measure RX latency (ms per codeword)
    """
    encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
    channel = AWGN()
    no = ebnodb2no(tf.constant([TEST_SNR_DB], tf.float32), num_bits_per_symbol, coderate)
    no = expand_to_rank(no, 2)
    
    results = {}
    mapper = Mapper(constellation=Constellation("qam", num_bits_per_symbol, trainable=False))
    
    for batch_size in batch_sizes:
        times = []
        print(f"  [RX] 开始 batch_size={batch_size}, codewords={num_codewords}, trials={num_trials}")
        
        for trial in range(num_trials):
            num_batches = max(1, num_codewords // batch_size)
            
            # Generate received symbols
            b = tf.cast(tf.random.uniform((batch_size, k), minval=0, maxval=2, dtype=tf.int32), tf.float32)
            c = encoder(b)
            x = mapper(c)
            y = channel(x, no)
            ebno_db = tf.fill([batch_size], TEST_SNR_DB)
            
            # Warmup
            _ = rx_model(batch_size=batch_size, y=y, ebno_db=ebno_db)
            
            # Benchmark
            t_start = time.perf_counter()
            report_every = max(1, num_batches // 5)
            for bi in range(num_batches):
                # 每轮刷新输入，避免仅测同一帧
                b = tf.cast(tf.random.uniform((batch_size, k), minval=0, maxval=2, dtype=tf.int32), tf.float32)
                c = encoder(b)
                x = mapper(c)
                y = channel(x, no)
                _ = rx_model(batch_size=batch_size, y=y, ebno_db=ebno_db)

                if progress and ((bi + 1) % report_every == 0 or (bi + 1) == num_batches):
                    elapsed = time.perf_counter() - t_start
                    done = bi + 1
                    print(f"    [RX][bs={batch_size}][trial={trial+1}/{num_trials}] {done}/{num_batches} batches, elapsed={elapsed:.1f}s")
            t_end = time.perf_counter()
            
            latency_ms = (t_end - t_start) / (batch_size * num_batches) * 1000  # ms/codeword
            times.append(latency_ms)
        
        results[batch_size] = {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "times": times,
        }
        print(f"  RX Batch {batch_size:3d}: {np.mean(times):8.3f} ± {np.std(times):6.3f} ms/codeword")
    
    return results


def benchmark_e2e_ber(tx_model, rx_model, snr_db_values=[4.0, 5.0, 6.0, 7.0, 8.0], num_codewords=1000):
    """
    Measure end-to-end BER across SNR range
    """
    encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
    channel = AWGN()
    
    results = {}
    batch_size = 64
    
    for snr_db in snr_db_values:
        ber_list = []
        no = ebnodb2no(tf.constant([snr_db], tf.float32), num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)
        ebno_db = tf.fill([batch_size], snr_db)
        
        num_batches = num_codewords // batch_size
        num_errors = 0
        num_total = 0
        
        for _ in range(num_batches):
            b = tf.cast(tf.random.uniform((batch_size, k), minval=0, maxval=2, dtype=tf.int32), tf.float32)
            c = encoder(b)
            x = tx_model(batch_size=batch_size, c=c, ebno_db=ebno_db)
            y = channel(x, no)
            b_hat = rx_model(batch_size=batch_size, y=y, ebno_db=ebno_db)
            
            # Hard decision and BER computation
            b_hat_hard = tf.cast(b_hat > 0.5, tf.float32)
            num_errors += tf.reduce_sum(tf.cast(tf.not_equal(b, b_hat_hard), tf.int32)).numpy()
            num_total += batch_size * k
        
        ber = num_errors / num_total if num_total > 0 else 0.0
        results[float(snr_db)] = ber
        print(f"  SNR {snr_db:.1f} dB: BER = {ber:.6f}")
    
    return results


# ==================== Main Benchmark ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="快速模式：更少码字、更少trial")
    parser.add_argument("--tx_codewords", type=int, default=NUM_TEST_CODEWORDS, help="TX 吞吐测试码字数")
    parser.add_argument("--rx_codewords", type=int, default=NUM_RX_TEST_CODEWORDS, help="RX 时延测试码字数")
    parser.add_argument("--ber_codewords", type=int, default=1000, help="BER 测试码字数")
    parser.add_argument("--trials", type=int, default=NUM_TRIALS, help="每个batch重复次数")
    parser.add_argument("--no_progress", action="store_true", help="关闭RX阶段进度打印")
    parser.add_argument("--neural_ckpt_path", type=str, default="", help="神经调制/解调 checkpoint 路径，默认优先 ckpt-0")
    parser.add_argument("--neural_ckpt_dir", type=str, default="neural_modem_checkpoints_v2", help="神经 checkpoint 目录")
    parser.add_argument("--neural_llr_sign", type=float, default=1.0, help="神经解调LLR符号，默认+1")
    args = parser.parse_args()

    tx_codewords = args.tx_codewords
    rx_codewords = args.rx_codewords
    ber_codewords = args.ber_codewords
    trials = args.trials
    show_progress = not args.no_progress

    if args.quick:
        tx_codewords = min(tx_codewords, 512)
        rx_codewords = min(rx_codewords, 256)
        ber_codewords = min(ber_codewords, 256)
        trials = min(trials, 1)

    if args.neural_ckpt_path:
        neural_ckpt_path = args.neural_ckpt_path
    else:
        preferred = os.path.join(args.neural_ckpt_dir, "ckpt-0")
        if tf.io.gfile.exists(preferred + ".index"):
            neural_ckpt_path = preferred
        else:
            neural_ckpt_path = tf.train.latest_checkpoint(args.neural_ckpt_dir) or ""

    print(f"[INFO] Neural checkpoint: {neural_ckpt_path if neural_ckpt_path else '<none>'}")

    print("\n" + "="*80)
    print("NEURAL MODULATOR vs TRADITIONAL BASELINE - COMPREHENSIVE BENCHMARK")
    print("="*80)
    
    # Create results directory
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "num_bits_per_symbol": num_bits_per_symbol,
            "coderate": coderate,
            "n": n,
            "k": k,
            "test_batch_sizes": TEST_BATCH_SIZES,
            "test_snr_db": TEST_SNR_DB,
            "tx_codewords": tx_codewords,
            "rx_codewords": rx_codewords,
            "ber_codewords": ber_codewords,
            "num_trials": trials,
            "quick_mode": args.quick,
            "neural_ckpt_path": neural_ckpt_path,
        },
        "results": {},
    }
    
    # ===== TRADITIONAL BASELINE =====
    print("\n[PHASE 1] Testing TRADITIONAL QAM + APP Demapper")
    print("-" * 80)
    
    tx_trad = TraditionalTXPhysical()
    rx_trad = TraditionalRXPhysical()
    
    print("\nTX Throughput (Traditional):")
    tx_trad_results = benchmark_tx_throughput(tx_trad, TEST_BATCH_SIZES, tx_codewords, trials)
    
    print("\nRX Latency (Traditional):")
    rx_trad_results = benchmark_rx_latency(rx_trad, TEST_BATCH_SIZES, rx_codewords, trials, progress=show_progress)
    
    print("\nEnd-to-End BER (Traditional):")
    ber_trad_results = benchmark_e2e_ber(tx_trad, rx_trad, num_codewords=ber_codewords)
    
    all_results["results"]["traditional"] = {
        "tx_throughput": tx_trad_results,
        "rx_latency": rx_trad_results,
        "ber": ber_trad_results,
    }
    
    # ===== NEURAL BACKEND =====
    if NEURAL_AVAILABLE:
        print("\n[PHASE 2] Testing NEURAL Modulator + Neural Demapper")
        print("-" * 80)
        
        try:
            tx_neural = NeuralTXPhysical(ckpt_dir=args.neural_ckpt_dir, ckpt_path=neural_ckpt_path)
            rx_neural = NeuralRXPhysical(
                ckpt_dir=args.neural_ckpt_dir,
                ckpt_path=neural_ckpt_path,
                llr_sign=args.neural_llr_sign,
            )
            
            print("\nTX Throughput (Neural):")
            tx_neural_results = benchmark_tx_throughput(tx_neural, TEST_BATCH_SIZES, tx_codewords, trials)
            
            print("\nRX Latency (Neural):")
            rx_neural_results = benchmark_rx_latency(rx_neural, TEST_BATCH_SIZES, rx_codewords, trials, progress=show_progress)
            
            print("\nEnd-to-End BER (Neural):")
            ber_neural_results = benchmark_e2e_ber(tx_neural, rx_neural, num_codewords=ber_codewords)
            
            all_results["results"]["neural"] = {
                "tx_throughput": tx_neural_results,
                "rx_latency": rx_neural_results,
                "ber": ber_neural_results,
            }
            
            # ===== COMPARATIVE ANALYSIS =====
            print("\n[PHASE 3] Comparative Analysis")
            print("-" * 80)
            
            print("\nThroughput Improvement (Neural vs Traditional):")
            for bs in TEST_BATCH_SIZES:
                trad_tp = tx_trad_results[bs]["mean"]
                neural_tp = tx_neural_results[bs]["mean"]
                improvement = (neural_tp - trad_tp) / trad_tp * 100
                print(f"  Batch {bs:3d}: {improvement:+7.2f}% ({trad_tp:.1f} → {neural_tp:.1f} codewords/sec)")
            
            print("\nLatency Reduction (Neural vs Traditional):")
            for bs in TEST_BATCH_SIZES:
                trad_lat = rx_trad_results[bs]["mean"]
                neural_lat = rx_neural_results[bs]["mean"]
                reduction = (trad_lat - neural_lat) / trad_lat * 100
                print(f"  Batch {bs:3d}: {reduction:+7.2f}% ({trad_lat:.3f} → {neural_lat:.3f} ms/codeword)")
            
            print("\nBER Comparison:")
            for snr in sorted(ber_trad_results.keys()):
                ber_trad = ber_trad_results[snr]
                ber_neural = ber_neural_results[snr]
                delta = (ber_neural - ber_trad) / ber_trad * 100 if ber_trad > 0 else 0
                print(f"  SNR {snr:.1f} dB: Traditional={ber_trad:.6f}, Neural={ber_neural:.6f} ({delta:+.2f}%)")
            
        except Exception as e:
            print(f"[ERROR] Failed to benchmark neural backend: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[SKIP] Neural backend not available")
    
    # ===== Save Results =====
    print("\n[SAVING] Results to JSON and plots...")
    
    results_file = results_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    # ===== Plot Results =====
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # TX Throughput
        ax = axes[0, 0]
        if "traditional" in all_results["results"]:
            trad_bss = sorted(all_results["results"]["traditional"]["tx_throughput"].keys())
            trad_tps = [all_results["results"]["traditional"]["tx_throughput"][bs]["mean"] for bs in trad_bss]
            ax.plot(trad_bss, trad_tps, 'o-', label='Traditional', linewidth=2, markersize=6)
        
        if "neural" in all_results["results"]:
            neural_bss = sorted(all_results["results"]["neural"]["tx_throughput"].keys())
            neural_tps = [all_results["results"]["neural"]["tx_throughput"][bs]["mean"] for bs in neural_bss]
            ax.plot(neural_bss, neural_tps, 's-', label='Neural', linewidth=2, markersize=6)
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Throughput (codewords/sec)')
        ax.set_title('TX Throughput vs Batch Size')
        ax.set_xscale('log', base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # RX Latency
        ax = axes[0, 1]
        if "traditional" in all_results["results"]:
            trad_bss = sorted(all_results["results"]["traditional"]["rx_latency"].keys())
            trad_lats = [all_results["results"]["traditional"]["rx_latency"][bs]["mean"] for bs in trad_bss]
            ax.plot(trad_bss, trad_lats, 'o-', label='Traditional', linewidth=2, markersize=6)
        
        if "neural" in all_results["results"]:
            neural_bss = sorted(all_results["results"]["neural"]["rx_latency"].keys())
            neural_lats = [all_results["results"]["neural"]["rx_latency"][bs]["mean"] for bs in neural_bss]
            ax.plot(neural_bss, neural_lats, 's-', label='Neural', linewidth=2, markersize=6)
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Latency (ms/codeword)')
        ax.set_title('RX Latency vs Batch Size')
        ax.set_xscale('log', base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # BER
        ax = axes[1, 0]
        if "traditional" in all_results["results"]:
            snrs = sorted(all_results["results"]["traditional"]["ber"].keys())
            bers = [all_results["results"]["traditional"]["ber"][s] for s in snrs]
            ax.semilogy(snrs, bers, 'o-', label='Traditional', linewidth=2, markersize=6)
        
        if "neural" in all_results["results"]:
            snrs = sorted(all_results["results"]["neural"]["ber"].keys())
            bers = [all_results["results"]["neural"]["ber"][s] for s in snrs]
            ax.semilogy(snrs, bers, 's-', label='Neural', linewidth=2, markersize=6)
        
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('BER')
        ax.set_title('End-to-End BER vs SNR')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        # Speedup factor
        ax = axes[1, 1]
        if "traditional" in all_results["results"] and "neural" in all_results["results"]:
            bss = sorted(all_results["results"]["traditional"]["rx_latency"].keys())
            speedups = []
            for bs in bss:
                trad_lat = all_results["results"]["traditional"]["rx_latency"][bs]["mean"]
                neural_lat = all_results["results"]["neural"]["rx_latency"][bs]["mean"]
                speedup = trad_lat / neural_lat
                speedups.append(speedup)
            
            ax.plot(bss, speedups, 'o-', label='RX Latency Speedup', linewidth=2, markersize=6, color='green')
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Speedup Factor (Traditional / Neural)')
            ax.set_title('RX Latency Speedup')
            ax.set_xscale('log', base=2)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = results_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=150)
        print(f"Plot saved to: {plot_file}")
        
    except Exception as e:
        print(f"[WARN] Failed to generate plots: {e}")
    
    print("\n" + "="*80)
    print("Benchmark completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

