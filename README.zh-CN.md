# EdgeSemCom Infra：端侧语义通信基础设施

[English](README.md) · [架构说明](docs/architecture.md) ·
[复现实验](docs/reproduction.md) · [实验结果](docs/results.md)

EdgeSemCom Infra 将语义表征学习、端侧推理和软件无线电整合为一条端到端图像
通信链路。从模型导出、Jetson Orin 推理优化，到基带处理和 USRP X310 真实空口
评估，仓库提供可运行的工程模块、实验入口与结果记录。

![系统架构](assets/system-architecture.png)

## 核心内容

- **AI Infra：** VAE 的 TensorRT/ONNX 导出路径、TensorFlow 图执行、批处理基带
  运算、端侧性能测量与回退机制。
- **语义链路：** VAE 潜变量、Top-K 选择、分位数量化、LDPC 保护和图像重构。
- **神经物理层：** 神经调制/软解调、可训练星座和传统/神经模块分阶段消融。
- **真实链路适配：** 导频同步、LS/LMMSE-DFT 信道估计、迭代 MMSE 均衡和
  可学习残差补偿。
- **跨工具部署：** 将训练星座导出到 MATLAB，并提供查表调制与最近邻解调。

## 代表性结果

以下为 Jetson Orin 部署与 USRP X310 空口实验的代表性结果；分阶段数据和实验
配置见[详细结果](docs/results.md)。

| 指标 | 优化前 | 优化后/观测值 | 变化 |
|---|---:|---:|---:|
| Jetson 端到端时延 | 3464.9 ms | 639.8 ms | **降低 81.54%** |
| TX：LDPC 编码与调制 | 88.3 ms | 7.4 ms | **降低 91.62%** |
| RX：解调与 LDPC 译码 | 1380.9 ms | 110.0 ms | **降低 92.03%** |
| 真实空口图像质量均值 | — | 28.10 dB PSNR / 0.982 SSIM | — |

## 快速开始

仿真环境用于训练和组件评估；Jetson 部署使用与 JetPack 兼容的 CUDA、
TensorRT、PyTorch 和 TensorFlow 运行环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[simulation,dev]"
pytest

python training/train_neural_phy.py --epochs 40 --save_dir artifacts/neural-phy
python evaluation/compare_phy_layers.py \
  --neural_model artifacts/neural-phy/best_model.pt \
  --output_dir results/generated/phy-comparison
```

端侧部署、模型配置和空口实验步骤见[复现指南](docs/reproduction.md)。

## 开源许可

代码采用 [MIT License](LICENSE)；第三方框架与硬件 SDK 遵循各自许可证。

