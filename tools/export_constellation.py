#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
星座图导出脚本：从 Sionna 训练模型中提取已训练的 64-QAM 星座，导出为 MATLAB 可读格式。

使用方式：
    # 从 checkpoint 导出星座
    python export_trained_constellation.py \\
        --ckpt_dir checkpoints/awgn_training_v1 \\
        --output trained_constellation.mat \\
        --format mat
    
    # 或导出为 JSON
    python export_trained_constellation.py \\
        --model_weights weights.pkl \\
        --output trained_constellation.json \\
        --format json

输出格式：
    MAT 格式: constellation 变量 shape=[64, 2] (real/imag 分开)
    JSON 格式: {"real": [...], "imag": [...]}
    C 代码: C 数组初始化
    MATLAB 代码: qammod() 等效查表生成器
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
tf.get_logger().setLevel("ERROR")

# Sionna 导入（与 deep_tx.py 保持一致）
try:
    import sionna.phy
    from sionna.phy.mapping import Constellation
except ImportError as e:
    print(f"[ERROR] Failed to import Sionna: {e}")
    sys.exit(1)


def load_constellation_from_checkpoint(ckpt_dir: str):
    """
    从 TensorFlow checkpoint 加载已训练的星座参数。
    
    Args:
        ckpt_dir: 包含 checkpoint 的目录 (e.g., checkpoints/awgn_training_v1)
    
    Returns:
        constellation_complex: [64,] 复数型星座点
        points_r: [64,] 实部
        points_i: [64,] 虚部
    """
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if not latest_ckpt:
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_dir}")
    
    print(f"[INFO] 加载 checkpoint: {latest_ckpt}")
    
    # 重建模型类（与 deep_tx.py 一致）
    from deep_tx import TXSystemConventionalTraining
    
    model = TXSystemConventionalTraining(training=False)
    
    # Dummy call 以构建权重
    dummy_batch = tf.constant(1, tf.int32)
    dummy_bits = tf.zeros([1, 15000], dtype=tf.float32)  # k=7500, num_bits=6 => 45000 bits
    dummy_ebno = tf.constant(5.5, tf.float32)
    _ = model(dummy_batch, dummy_bits, dummy_ebno)
    
    # 加载权重
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(latest_ckpt).expect_partial()
    
    points_r = model.points_r.numpy()
    points_i = model.points_i.numpy()
    constellation = (points_r + 1j * points_i).astype(np.complex64)
    
    print(f"[INFO] 提取星座 shape: {constellation.shape}")
    print(f"[INFO] 星座实部范围: [{points_r.min():.4f}, {points_r.max():.4f}]")
    print(f"[INFO] 星座虚部范围: [{points_i.min():.4f}, {points_i.max():.4f}]")
    
    return constellation, points_r, points_i


def load_constellation_from_pickle(weights_path: str):
    """
    从 pickle 权重文件（旧格式）加载星座。
    
    Args:
        weights_path: 权重 pickle 文件路径
    
    Returns:
        constellation_complex: [64,] 复数型星座点
        points_r: [64,] 实部
        points_i: [64,] 虚部
    """
    print(f"[INFO] 从 pickle 加载权重: {weights_path}")
    
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    
    # weights 是列表，第一个元素通常是 points_r，第二个是 points_i
    if isinstance(weights, list) and len(weights) >= 2:
        points_r = np.array(weights[0])
        points_i = np.array(weights[1])
    else:
        raise ValueError("Pickle 格式不符合预期，需要 [points_r, points_i]")
    
    constellation = (points_r + 1j * points_i).astype(np.complex64)
    
    print(f"[INFO] 提取星座 shape: {constellation.shape}")
    print(f"[INFO] 星座实部范围: [{points_r.min():.4f}, {points_r.max():.4f}]")
    print(f"[INFO] 星座虚部范围: [{points_i.min():.4f}, {points_i.max():.4f}]")
    
    return constellation, points_r, points_i


def export_to_mat(constellation, points_r, points_i, output_path: str):
    """导出为 MATLAB MAT 格式。"""
    try:
        from scipy.io import savemat
    except ImportError:
        print("[ERROR] scipy 未安装，无法导出 MAT 格式")
        return False
    
    # 构建 MAT 字典
    mat_dict = {
        'constellation': constellation,        # 复数型
        'real': points_r,                       # 实部 [64,]
        'imag': points_i,                       # 虚部 [64,]
        'num_bits_per_symbol': np.array([6]),  # 常数
        'modulation_order': np.array([64]),     # 常数
    }
    
    savemat(output_path, mat_dict)
    print(f"[SUCCESS] MAT 文件已保存: {output_path}")
    print(f"[INFO] MATLAB 中使用: constellation = load('{output_path}');")
    return True


def export_to_json(constellation, points_r, points_i, output_path: str):
    """导出为 JSON 格式。"""
    json_dict = {
        "real": points_r.tolist(),
        "imag": points_i.tolist(),
        "modulation_order": 64,
        "num_bits_per_symbol": 6,
        "dtype": "float32",
    }
    
    with open(output_path, 'w') as f:
        json.dump(json_dict, f, indent=2)
    
    print(f"[SUCCESS] JSON 文件已保存: {output_path}")
    print(f"[INFO] MATLAB 中使用: jsondecoder.decode(...)")
    return True


def export_to_npy(constellation, points_r, points_i, output_path: str):
    """导出为 NumPy NPY 格式（可在 Python 中直接加载）。"""
    np.save(output_path.replace('.npy', '_real.npy'), points_r)
    np.save(output_path.replace('.npy', '_imag.npy'), points_i)
    
    print(f"[SUCCESS] NPY 文件已保存:")
    print(f"  - {output_path.replace('.npy', '_real.npy')}")
    print(f"  - {output_path.replace('.npy', '_imag.npy')}")
    return True


def export_to_c_header(constellation, points_r, points_i, output_path: str):
    """导出为 C 头文件（64 元数组）。"""
    with open(output_path, 'w') as f:
        f.write("// 自动生成的 64-QAM 训练星座\n")
        f.write("// 在 MATLAB/C/Python 中使用\n\n")
        
        f.write("#ifndef TRAINED_CONSTELLATION_H\n")
        f.write("#define TRAINED_CONSTELLATION_H\n\n")
        
        f.write("// 星座实部\n")
        f.write("const float CONSTELLATION_REAL[64] = {\n")
        for i in range(0, 64, 8):
            line = "  " + ", ".join(f"{points_r[j]:10.6f}" for j in range(i, min(i+8, 64)))
            f.write(line + (",\n" if i+8 < 64 else "\n"))
        f.write("};\n\n")
        
        f.write("// 星座虚部\n")
        f.write("const float CONSTELLATION_IMAG[64] = {\n")
        for i in range(0, 64, 8):
            line = "  " + ", ".join(f"{points_i[j]:10.6f}" for j in range(i, min(i+8, 64)))
            f.write(line + (",\n" if i+8 < 64 else "\n"))
        f.write("};\n\n")
        
        f.write("#endif // TRAINED_CONSTELLATION_H\n")
    
    print(f"[SUCCESS] C 头文件已保存: {output_path}")
    return True


def export_to_matlab_m(constellation, points_r, points_i, output_path: str):
    """导出为 MATLAB .m 函数，直接返回星座。"""
    with open(output_path, 'w') as f:
        f.write("""% 自动生成的训练星座查表函数
% 用法: constellation = trained_constellation()

function constellation = trained_constellation()
    % 从 Python 训练导出的 64-QAM 星座
    real_part = [...
""")
        for i in range(0, 64, 8):
            line = "        " + " ".join(f"{points_r[j]:10.6f}" for j in range(i, min(i+8, 64)))
            f.write(line + ("\n" if i+8 < 64 else " ...\n"))
        
        f.write("    ];\n\n")
        f.write("    imag_part = [...\n")
        
        for i in range(0, 64, 8):
            line = "        " + " ".join(f"{points_i[j]:10.6f}" for j in range(i, min(i+8, 64)))
            f.write(line + ("\n" if i+8 < 64 else " ...\n"))
        
        f.write("    ];\n\n")
        f.write("    constellation = complex(real_part, imag_part);\nend\n")
    
    print(f"[SUCCESS] MATLAB 函数已保存: {output_path}")
    return True


def print_summary(constellation, points_r, points_i):
    """打印星座统计信息。"""
    print("\n" + "="*60)
    print("星座统计信息:")
    print("="*60)
    print(f"调制阶数: 64 (6 bits/symbol)")
    print(f"星座点数: {len(constellation)}")
    print(f"\n实部统计:")
    print(f"  范围: [{points_r.min():.4f}, {points_r.max():.4f}]")
    print(f"  均值: {points_r.mean():.4f}")
    print(f"  标准差: {points_r.std():.4f}")
    print(f"\n虚部统计:")
    print(f"  范围: [{points_i.min():.4f}, {points_i.max():.4f}]")
    print(f"  均值: {points_i.mean():.4f}")
    print(f"  标准差: {points_i.std():.4f}")
    print(f"\n平均功率: {np.mean(np.abs(constellation)**2):.4f}")
    print(f"峰值功率: {np.max(np.abs(constellation)**2):.4f}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="从 Sionna 训练模型导出 64-QAM 星座"
    )
    
    # 输入源（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--ckpt_dir',
        help='TensorFlow checkpoint 目录 (e.g., checkpoints/awgn_training_v1)'
    )
    input_group.add_argument(
        '--model_weights',
        help='Pickle 权重文件路径 (e.g., awgn_autoencoder_weights_conventional_training_tx)'
    )
    
    # 输出
    parser.add_argument(
        '--output', '-o',
        default='trained_constellation',
        help='输出文件名（不含扩展名，默认: trained_constellation）'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['mat', 'json', 'npy', 'c', 'm'],
        default='mat',
        help='输出格式 (默认: mat)'
    )
    
    args = parser.parse_args()
    
    # 加载星座
    try:
        if args.ckpt_dir:
            constellation, points_r, points_i = load_constellation_from_checkpoint(args.ckpt_dir)
        else:
            constellation, points_r, points_i = load_constellation_from_pickle(args.model_weights)
    except Exception as e:
        print(f"[ERROR] 加载星座失败: {e}")
        sys.exit(1)
    
    # 打印统计
    print_summary(constellation, points_r, points_i)
    
    # 导出
    output_path = args.output
    if args.format == 'mat':
        if not output_path.endswith('.mat'):
            output_path += '.mat'
        export_to_mat(constellation, points_r, points_i, output_path)
    
    elif args.format == 'json':
        if not output_path.endswith('.json'):
            output_path += '.json'
        export_to_json(constellation, points_r, points_i, output_path)
    
    elif args.format == 'npy':
        export_to_npy(constellation, points_r, points_i, output_path)
    
    elif args.format == 'c':
        if not output_path.endswith('.h'):
            output_path += '.h'
        export_to_c_header(constellation, points_r, points_i, output_path)
    
    elif args.format == 'm':
        if not output_path.endswith('.m'):
            output_path += '.m'
        export_to_matlab_m(constellation, points_r, points_i, output_path)
    
    print(f"\n[SUCCESS] 导出完成！")


if __name__ == '__main__':
    main()

