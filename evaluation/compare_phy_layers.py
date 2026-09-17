#!/usr/bin/env python3
"""
神经网络物理层 vs Sionna 物理层对比评估

评估指标：
- BER (比特错误率)
- FER (帧错误率)  
- BLER (块错误率)
- 延迟 (推理时间)
- 功率效率
"""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from edge_semcom.neural_phy import NeuralPhysicalLayer
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Constellation, BinarySource, Demapper
from sionna.phy.utils import ebnodb2no, expand_to_rank, log10
import sionna.phy


class SionnaPhysicalLayer(nn.Module):
    """Sionna 物理层包装"""
    def __init__(
        self,
        k: int = 7500,
        n: int = 15000,
        num_bits_per_symbol: int = 6,
        coderate: float = 0.5,
    ):
        super().__init__()
        self.k = k
        self.n = n
        self.num_bits_per_symbol = num_bits_per_symbol
        self.coderate = coderate
        
        # Sionna 组件
        self.encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)
        
        # 星座
        constellation = Constellation("qam", num_bits_per_symbol)
        self.mapper = Mapper(constellation=constellation)
        self.demapper = Demapper("app", constellation=constellation)
        
    def forward(
        self,
        bits: torch.Tensor,
        snr_db: torch.Tensor,
        channel_type: str = "awgn",
    ) -> Dict[str, torch.Tensor]:
        """端到端 Sionna 链路"""
        batch_size = bits.shape[0]
        device = bits.device
        
        # 编码
        bits_encoded = self.encoder(bits)  # [batch, n]
        
        # 调制
        symbols = self.mapper(bits_encoded)  # [batch, n/num_bits_per_symbol]
        
        # 通过信道
        rx_signal, h = self._transmit_through_channel(symbols, snr_db, channel_type)
        
        # 解调
        no = ebnodb2no(snr_db, self.num_bits_per_symbol, self.coderate)
        no = expand_to_rank(no, 2)
        
        llr = self.demapper(rx_signal / (h + 1e-8), no)  # [batch, n]
        
        # 解码
        bits_decoded = self.decoder(llr)  # [batch, k]
        
        return {
            "tx_symbols": symbols,
            "rx_signal": rx_signal,
            "h_true": h,
            "bits_decoded": bits_decoded,
        }
    
    def _transmit_through_channel(
        self,
        symbols: torch.Tensor,
        snr_db: torch.Tensor,
        channel_type: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """模拟信道"""
        batch_size = symbols.shape[0]
        
        # 信道衰减
        if channel_type == "awgn":
            h = torch.ones_like(symbols)
        elif channel_type == "rayleigh":
            h_real = torch.randn_like(symbols.real) / np.sqrt(2)
            h_imag = torch.randn_like(symbols.imag) / np.sqrt(2)
            h = torch.complex(h_real, h_imag)
        else:
            h = torch.ones_like(symbols)
        
        # 接收
        rx_noisy = h * symbols
        
        # 噪声
        snr_linear = 10 ** (snr_db / 10)
        noise_power = 1.0 / snr_linear
        
        noise_real = torch.randn_like(rx_noisy.real) * torch.sqrt(noise_power / 2).unsqueeze(-1)
        noise_imag = torch.randn_like(rx_noisy.imag) * torch.sqrt(noise_power / 2).unsqueeze(-1)
        noise = torch.complex(noise_real, noise_imag)
        
        rx_signal = rx_noisy + noise
        
        return rx_signal, h


class PhysicalLayerComparator:
    """物理层对比工具"""
    def __init__(self, device: torch.device):
        self.device = device
        
    def ber_curve(
        self,
        model_name: str,
        model: nn.Module,
        snr_range: np.ndarray,
        num_samples: int = 1000,
        k: int = 7500,
    ) -> Dict[float, float]:
        """计算 BER 曲线"""
        ber_dict = {}
        
        print(f"\n计算 {model_name} 的 BER 曲线...")
        
        for snr in snr_range:
            model.eval()
            total_errors = 0
            total_bits = 0
            
            with torch.no_grad():
                for _ in range(num_samples // 32):
                    bits = torch.randint(0, 2, (32, k), dtype=torch.float32).to(self.device)
                    snr_tensor = torch.tensor([snr], dtype=torch.float32).to(self.device)
                    
                    # 前向传播
                    output = model(bits, snr_tensor)
                    bits_decoded = output["bits_decoded"]
                    
                    # 计算错误
                    errors = torch.sum((bits_decoded != bits)).item()
                    total_errors += errors
                    total_bits += bits.numel()
            
            ber = total_errors / total_bits
            ber_dict[float(snr)] = ber
            print(f"  SNR={snr:.1f}dB: BER={ber:.6f}")
        
        return ber_dict
    
    def latency_benchmark(
        self,
        model_name: str,
        model: nn.Module,
        k: int = 7500,
        num_trials: int = 100,
    ) -> Dict[str, float]:
        """延迟基准测试"""
        print(f"\n性能基准: {model_name}")
        
        model.eval()
        latencies = []
        
        with torch.no_grad():
            # 预热
            for _ in range(10):
                bits = torch.randint(0, 2, (1, k), dtype=torch.float32).to(self.device)
                snr = torch.tensor([10.0], dtype=torch.float32).to(self.device)
                _ = model(bits, snr)
            
            # 测量
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            for _ in range(num_trials):
                bits = torch.randint(0, 2, (1, k), dtype=torch.float32).to(self.device)
                snr = torch.tensor([10.0], dtype=torch.float32).to(self.device)
                
                start_time = time.time()
                _ = model(bits, snr)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                latencies.append((end_time - start_time) * 1000)  # ms
        
        latencies = np.array(latencies)
        
        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
        }
    
    def compare_all(
        self,
        neural_model_path: str,
        snr_range: np.ndarray = np.arange(0, 15, 1),
        k: int = 7500,
    ) -> Dict:
        """完整对比"""
        
        # 初始化模型
        neural_model = NeuralPhysicalLayer(k=k, n_symbols=k//3)
        neural_model.load(neural_model_path)
        neural_model = neural_model.to(self.device)
        
        sionna_model = SionnaPhysicalLayer(k=k)
        sionna_model = sionna_model.to(self.device)
        
        # 对比结果
        results = {
            "config": {
                "k": k,
                "snr_range": snr_range.tolist(),
            },
            "neural": {},
            "sionna": {},
        }
        
        # BER 对比
        results["neural"]["ber_curve"] = self.ber_curve(
            "神经网络物理层",
            neural_model,
            snr_range,
            k=k
        )
        
        results["sionna"]["ber_curve"] = self.ber_curve(
            "Sionna LDPC",
            sionna_model,
            snr_range,
            k=k
        )
        
        # 延迟对比
        results["neural"]["latency"] = self.latency_benchmark("神经网络物理层", neural_model, k=k)
        results["sionna"]["latency"] = self.latency_benchmark("Sionna LDPC", sionna_model, k=k)
        
        return results
    
    def plot_comparison(self, results: Dict, save_path: str = "comparison.png"):
        """绘制对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # BER 曲线
        neural_ber = results["neural"]["ber_curve"]
        sionna_ber = results["sionna"]["ber_curve"]
        
        snr_vals = sorted(neural_ber.keys())
        neural_ber_vals = [neural_ber[s] for s in snr_vals]
        sionna_ber_vals = [sionna_ber[s] for s in snr_vals]
        
        axes[0].semilogy(snr_vals, neural_ber_vals, 'o-', label='神经网络物理层', linewidth=2)
        axes[0].semilogy(snr_vals, sionna_ber_vals, 's-', label='Sionna LDPC', linewidth=2)
        axes[0].set_xlabel('SNR (dB)', fontsize=12)
        axes[0].set_ylabel('BER', fontsize=12)
        axes[0].set_title('BER 对比', fontsize=14)
        axes[0].grid(True, which='both', alpha=0.3)
        axes[0].legend()
        
        # 延迟对比
        models = ['神经网络', 'Sionna']
        latencies = [
            results["neural"]["latency"]["mean_ms"],
            results["sionna"]["latency"]["mean_ms"],
        ]
        
        axes[1].bar(models, latencies, color=['blue', 'orange'], alpha=0.7)
        axes[1].set_ylabel('延迟 (ms)', fontsize=12)
        axes[1].set_title('推理延迟对比', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(latencies):
            axes[1].text(i, v + 0.5, f'{v:.2f}ms', ha='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 对比图已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="物理层对比评估")
    parser.add_argument("--neural_model", type=str, required=True, help="神经网络模型路径")
    parser.add_argument("--snr_min", type=float, default=0, help="最小 SNR")
    parser.add_argument("--snr_max", type=float, default=15, help="最大 SNR")
    parser.add_argument("--k", type=int, default=7500, help="信息比特数")
    parser.add_argument("--output_dir", type=str, default="comparison_results", help="输出目录")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建对比工具
    comparator = PhysicalLayerComparator(device)
    
    # 执行对比
    snr_range = np.arange(args.snr_min, args.snr_max + 1, 1)
    results = comparator.compare_all(args.neural_model, snr_range=snr_range, k=args.k)
    
    # 保存结果
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    result_path = Path(args.output_dir) / "comparison_results.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 结果已保存: {result_path}")
    
    # 绘制对比图
    plot_path = Path(args.output_dir) / "comparison.png"
    comparator.plot_comparison(results, save_path=str(plot_path))
    
    # 打印总结
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)
    print(f"\n神经网络物理层:")
    print(f"  平均延迟: {results['neural']['latency']['mean_ms']:.2f}ms")
    print(f"  最坏延迟: {results['neural']['latency']['max_ms']:.2f}ms")
    
    print(f"\nSionna LDPC:")
    print(f"  平均延迟: {results['sionna']['latency']['mean_ms']:.2f}ms")
    print(f"  最坏延迟: {results['sionna']['latency']['max_ms']:.2f}ms")
    
    speedup = results['sionna']['latency']['mean_ms'] / results['neural']['latency']['mean_ms']
    print(f"\n加速比: {speedup:.2f}x")


if __name__ == "__main__":
    main()

