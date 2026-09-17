#!/usr/bin/env python3
"""
神经网络物理层实现 - 用于替代 Sionna
包含端到端的神经编码/解码、调制解调、信道估计

工业级特性:
- 可配置的网络架构
- 分模块实现（易于维护和升级）
- 完整的训练管道
- 与 Sionna 物理层兼容的接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import json
from pathlib import Path


class ResidualBlock(nn.Module):
    """残差块 - 深度网络的基础构件"""
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.relu(self.fc1(x))
        x = self.norm2(x)
        x = self.fc2(x)
        return x + residual


class NeuralEncoder(nn.Module):
    """
    神经网络编码器 - 替代 Sionna 的 LDPC 编码器
    
    输入: 信息比特 [batch, k]
    输出: 编码后的符号 [batch, n_symbols]
    """
    def __init__(self, k: int = 7500, n_symbols: int = 2500, hidden_dim: int = 512):
        """
        Args:
            k: 信息比特数
            n_symbols: 输出符号数
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.k = k
        self.n_symbols = n_symbols
        
        # 编码网络：比特 → 复符号
        # 关键修复：扩大中间层，确保足够的信息容量处理 k 个输入比特
        self.encoder_net = nn.Sequential(
            nn.Linear(k, hidden_dim * 4),  # 7500 → 2048
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            ResidualBlock(hidden_dim * 4, hidden_dim * 8),  # 2048 内部扩展
            ResidualBlock(hidden_dim * 4, hidden_dim * 8),  # 残差块保持维度
            nn.Linear(hidden_dim * 4, n_symbols * 2),      # 2048 → 5000 (I/Q)
        )
        
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bits: [batch, k] 二进制比特
        
        Returns:
            symbols: [batch, n_symbols] 复符号
        """
        # 转换为浮点
        bits_float = bits.float()
        
        # 编码
        iq_output = self.encoder_net(bits_float)  # [batch, n_symbols*2]
        
        # 转换为复数并归一化功率
        iq_reshaped = iq_output.reshape(-1, self.n_symbols, 2)
        symbols = torch.complex(iq_reshaped[..., 0], iq_reshaped[..., 1])
        
        # 功率归一化 (平均功率 = 1)
        power = torch.mean(torch.abs(symbols) ** 2, dim=1, keepdim=True)
        symbols = symbols / torch.sqrt(power + 1e-8)
        
        return symbols


class ChannelEstimator(nn.Module):
    """
    神经网络信道估计器 - 替代 Sionna 的理想信道估计
    
    在实际应用中，估计多径信道
    """
    def __init__(self, n_symbols: int = 2500, hidden_dim: int = 128):
        super().__init__()
        self.n_symbols = n_symbols
        
        # 信道估计网络：接收信号 + SNR → 信道估计
        input_size = n_symbols * 3  # real(rx) + imag(rx) + snr
        self.estimator_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim, hidden_dim * 2),
            ResidualBlock(hidden_dim, hidden_dim * 2),
            nn.Linear(hidden_dim, n_symbols * 2),  # 输出复信道增益
        )
        
    def forward(self, rx_signal: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rx_signal: [batch, n_symbols] 接收复信号
            snr_db: [batch] SNR (dB)
        
        Returns:
            h_est: [batch, n_symbols] 估计的信道增益
        """
        batch_size = rx_signal.shape[0]
        n_symbols = rx_signal.shape[1]
        
        # 特征提取
        rx_real = torch.real(rx_signal)  # [batch, n_symbols]
        rx_imag = torch.imag(rx_signal)  # [batch, n_symbols]
        
        # 处理 SNR
        if snr_db.dim() == 0:
            snr_db = snr_db.unsqueeze(0)  # [1] -> [1]
        
        snr_linear = 10 ** (snr_db / 10)  # [batch] 
        snr_linear = snr_linear.unsqueeze(-1).expand(batch_size, 1)  # [batch, 1]
        snr_linear = snr_linear.expand(batch_size, n_symbols)  # [batch, n_symbols]
        
        features = torch.stack([rx_real, rx_imag, snr_linear], dim=-1)  # [batch, n_symbols, 3]
        features = features.reshape(batch_size, -1)  # [batch, n_symbols*3]
        
        # 估计
        h_iq = self.estimator_net(features)  # [batch, n_symbols*2]
        h_iq = h_iq.reshape(batch_size, n_symbols, 2)  # [batch, n_symbols, 2]
        h_est = torch.complex(h_iq[..., 0], h_iq[..., 1])  # [batch, n_symbols]
        
        return h_est


class NeuralDecoder(nn.Module):
    """
    神经网络解码器 - 替代 Sionna 的 LDPC 解码器
    
    输入: 接收信号 + 信道估计
    输出: 恢复的比特
    """
    def __init__(self, k: int = 7500, n_symbols: int = 2500, hidden_dim: int = 512):
        super().__init__()
        self.k = k
        self.n_symbols = n_symbols
        
        # 解码网络：接收的 IQ + 信道估计 → 比特
        # 关键修复：扩大中间层容量，从 5000 维（rx+h）中恢复 7500 个比特
        self.decoder_net = nn.Sequential(
            nn.Linear(n_symbols * 4, hidden_dim * 4),      # 10000 → 2048
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            ResidualBlock(hidden_dim * 4, hidden_dim * 8),  # 2048 内部扩展
            ResidualBlock(hidden_dim * 4, hidden_dim * 8),  # 残差块保持维度
            ResidualBlock(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 4, k),                   # 2048 → 7500 (比特 logits)
        )
        
        self.initialize_weights()
        
        # 最后一层输出缩小 (必须在所有初始化后)
        # 但不能缩小太多，否则模型无法学习
        with torch.no_grad():
            self.decoder_net[-1].weight.mul_(0.5)  # 0.5 而不是 0.05
            if self.decoder_net[-1].bias is not None:
                self.decoder_net[-1].bias.mul_(0.5)
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, rx_signal: torch.Tensor, h_est: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rx_signal: [batch, n_symbols] 接收复信号
            h_est: [batch, n_symbols] 信道估计
        
        Returns:
            bit_logits: [batch, k] 比特对数似然
            bits_hard: [batch, k] 硬判决比特
        """
        # 特征提取
        rx_real = torch.real(rx_signal)
        rx_imag = torch.imag(rx_signal)
        h_real = torch.real(h_est)
        h_imag = torch.imag(h_est)
        
        features = torch.cat([rx_real, rx_imag, h_real, h_imag], dim=-1)
        
        # 解码
        bit_logits = self.decoder_net(features)
        
        # 硬判决
        bits_hard = (bit_logits > 0).float()
        
        return bit_logits, bits_hard


class NeuralPhysicalLayer(nn.Module):
    """
    完整的神经网络物理层 - 端到端的编码-调制-信道-解调-解码
    
    与 Sionna 物理层兼容的接口
    """
    def __init__(
        self,
        k: int = 7500,
        n_symbols: int = 2500,
        encoder_hidden: int = 512,
        decoder_hidden: int = 512,
    ):
        super().__init__()
        
        self.k = k
        self.n_symbols = n_symbols
        
        # 编码+调制
        self.encoder = NeuralEncoder(k, n_symbols, encoder_hidden)
        
        # 信道估计
        self.channel_estimator = ChannelEstimator(n_symbols)
        
        # 解调+解码
        self.decoder = NeuralDecoder(k, n_symbols, decoder_hidden)
        
    def forward(
        self,
        bits: torch.Tensor,
        snr_db: torch.Tensor,
        channel_type: str = "awgn",
    ) -> Dict[str, torch.Tensor]:
        """
        端到端通信链路
        
        Args:
            bits: [batch, k] 输入比特
            snr_db: [batch] 或 标量，SNR in dB
            channel_type: "awgn" 或 "rayleigh"
        
        Returns:
            dict 包含：
            - tx_symbols: 发送符号
            - rx_signal: 接收信号
            - h_est: 信道估计
            - bit_logits: 解码的比特对数似然
            - bits_decoded: 硬判决比特
        """
        batch_size = bits.shape[0]
        device = bits.device
        
        # 处理 SNR 维度
        if snr_db.dim() == 0:
            snr_db = snr_db.unsqueeze(0).expand(batch_size)
        
        # 编码+调制
        tx_symbols = self.encoder(bits)  # [batch, n_symbols]
        
        # 通过信道
        rx_signal, h_true = self._transmit_through_channel(
            tx_symbols, snr_db, channel_type
        )
        
        # 信道估计
        h_est = self.channel_estimator(rx_signal, snr_db)
        
        # 解调+解码
        bit_logits, bits_decoded = self.decoder(rx_signal, h_est)
        
        return {
            "tx_symbols": tx_symbols,
            "rx_signal": rx_signal,
            "h_true": h_true,
            "h_est": h_est,
            "bit_logits": bit_logits,
            "bits_decoded": bits_decoded,
        }
    
    def _transmit_through_channel(
        self,
        tx_symbols: torch.Tensor,
        snr_db: torch.Tensor,
        channel_type: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """模拟信道传输"""
        batch_size = tx_symbols.shape[0]
        n_symbols = tx_symbols.shape[1]
        device = tx_symbols.device
        
        # 确保 SNR 的维度正确 [batch]
        if snr_db.dim() == 0:
            snr_db = snr_db.unsqueeze(0)
        if snr_db.shape[0] == 1:
            snr_db = snr_db.expand(batch_size)
        
        # 信道衰减
        if channel_type == "awgn":
            h = torch.ones_like(tx_symbols)
        elif channel_type == "rayleigh":
            # Rayleigh 衰减
            h_real = torch.randn_like(tx_symbols.real) / np.sqrt(2)
            h_imag = torch.randn_like(tx_symbols.imag) / np.sqrt(2)
            h = torch.complex(h_real, h_imag)
        else:
            h = torch.ones_like(tx_symbols)
        
        # 接收信号 = 信道 * 发送 + 噪声
        rx_noisy = h * tx_symbols
        
        # AWGN 噪声
        snr_linear = 10 ** (snr_db / 10)  # [batch]
        noise_power = 1.0 / snr_linear     # [batch]
        
        # 正确的广播：[batch, 1] 匹配 [batch, n_symbols]
        noise_std = torch.sqrt(noise_power / 2).reshape(-1, 1)  # [batch, 1]
        
        noise_real = torch.randn_like(rx_noisy.real) * noise_std
        noise_imag = torch.randn_like(rx_noisy.imag) * noise_std
        noise = torch.complex(noise_real, noise_imag)
        
        rx_signal = rx_noisy + noise
        
        return rx_signal, h
    
    def save(self, path: str):
        """保存模型"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"✓ 模型已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        self.load_state_dict(torch.load(path))
        print(f"✓ 模型已加载: {path}")


if __name__ == "__main__":
    # 测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = NeuralPhysicalLayer(k=7500, n_symbols=2500)
    model = model.to(device)
    
    # 随机输入
    batch_size = 4
    bits = torch.randint(0, 2, (batch_size, 7500), dtype=torch.float32).to(device)
    snr_db = torch.tensor([10.0], dtype=torch.float32).to(device)
    
    # 前向传播
    output = model(bits, snr_db, channel_type="awgn")
    
    print("=" * 60)
    print("神经网络物理层测试")
    print("=" * 60)
    print(f"TX 符号: {output['tx_symbols'].shape}")
    print(f"RX 信号: {output['rx_signal'].shape}")
    print(f"比特输出: {output['bits_decoded'].shape}")
    print(f"设备: {device}")
    print("=" * 60)

