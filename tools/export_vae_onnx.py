#!/usr/bin/env python3
"""
将 VAE encoder 和 decoder 分别导出为 ONNX
注意：不包含 vae_scale 缩放因子（会单独在推理时处理）
"""
import torch
from diffusers import AutoencoderKL
import os
import torch.nn as nn

print("[ONNX导出] VAE encoder & decoder 导出脚本")
print("=" * 80)

# 加载 VAE
print("\n[1] 加载 VAE 模型...")
vae = AutoencoderKL.from_pretrained('../train_vae/vae', local_files_only=True)
vae_scale = getattr(vae.config, "scaling_factor", 1.0)
print(f"    ✓ VAE scale factor: {vae_scale}")

# 创建输出目录
os.makedirs('onnx_models', exist_ok=True)


class VAEEncoderMeanWrapper(nn.Module):
    """与 vae.encode(x).latent_dist.mean 语义对齐（不含 scaling_factor 乘法）。"""
    def __init__(self, vae_model):
        super().__init__()
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

    def forward(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, _ = torch.chunk(moments, 2, dim=1)
        return mean


class VAEDecoderCoreWrapper(nn.Module):
    """与 vae.decode(z).sample 的核心对齐：post_quant_conv + decoder。
    说明：外部仍按现有代码先做 z/vae_scale，再喂入本 wrapper。
    """
    def __init__(self, vae_model):
        super().__init__()
        self.post_quant_conv = vae_model.post_quant_conv
        self.decoder = vae_model.decoder

    def forward(self, z):
        z = self.post_quant_conv(z)
        x = self.decoder(z)
        return x

# ============================================================================
# 导出 VAE Encoder（TX 端用）
# ============================================================================
print("\n[2] 导出 VAE Encoder...")

encoder = VAEEncoderMeanWrapper(vae).eval()

# dummy input: [1, 3, 512, 512]
dummy_input_encoder = torch.randn(1, 3, 512, 512)

torch.onnx.export(
    encoder,
    dummy_input_encoder,
    'onnx_models/vae_encoder.onnx',
    input_names=['img'],
    output_names=['latent_mean'],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes={'img': {0: 'batch'}, 'latent_mean': {0: 'batch'}},
    verbose=False,
)
print("    ✓ 导出成功: onnx_models/vae_encoder.onnx")

# ============================================================================
# 导出 VAE Decoder（RX 端用）
# ============================================================================
print("\n[3] 导出 VAE Decoder...")

decoder = VAEDecoderCoreWrapper(vae).eval()

# dummy input: [1, 4, 64, 64]
dummy_input_decoder = torch.randn(1, 4, 64, 64)

torch.onnx.export(
    decoder,
    dummy_input_decoder,
    'onnx_models/vae_decoder.onnx',
    input_names=['z'],
    output_names=['sample'],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes={'z': {0: 'batch'}, 'sample': {0: 'batch'}},
    verbose=False,
)
print("    ✓ 导出成功: onnx_models/vae_decoder.onnx")

print("\n" + "=" * 80)
print("[完成] 两个ONNX模型已导出到 onnx_models/ 目录")
print("    - vae_encoder.onnx:  TX端使用（3通道512x512 → 4通道64x64）")
print("    - vae_decoder.onnx:  RX端使用（4通道64x64 → 3通道512x512）")
print("\n[下一步] 用以下命令将ONNX转换为TensorRT engine：")
print("\n  # TX 端（Jetson编码）：")
print("  trtexec --onnx=onnx_models/vae_encoder.onnx \\")
print("    --saveEngine=vae_encoder.trt \\")
print("    --precision=fp16 --workspace=2048")
print("\n  # RX 端（Jetson解码）：")
print("  trtexec --onnx=onnx_models/vae_decoder.onnx \\")
print("    --saveEngine=vae_decoder.trt \\")
print("    --precision=fp16 --workspace=2048")
print("=" * 80)

