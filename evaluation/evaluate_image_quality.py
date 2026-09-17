#!/usr/bin/env python3
"""
语义通信图像质量评估工具
用于对比重构图像与原始图像的质量差异
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import json

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️ 警告: scikit-image 未安装，SSIM 不可用")
    print("  安装: pip install scikit-image")

def calculate_psnr(img1, img2):
    """计算 PSNR (Peak Signal-to-Noise Ratio)"""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    """计算 SSIM (Structural Similarity Index)"""
    if not SKIMAGE_AVAILABLE:
        return None
    
    # 转为灰度以简化计算
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    
    return ssim(gray1, gray2, data_range=255)

def calculate_mse(img1, img2):
    """计算 MSE (Mean Squared Error)"""
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def analyze_noise(img):
    """分析图像噪声特征"""
    # 转灰度
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 计算 Laplacian 方差（焦点度）
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 计算各通道标准差
    if len(img.shape) == 3:
        b_std = np.std(img[:,:,0])
        g_std = np.std(img[:,:,1])
        r_std = np.std(img[:,:,2])
        stds = {"B": b_std, "G": g_std, "R": r_std}
    else:
        stds = {"Gray": np.std(gray)}
    
    # 检测蓝粉噪点（高饱和度异常像素）
    if len(img.shape) == 3:
        # 转 HSV 检测异常像素
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 蓝色范围（100-130 in OpenCV HSV）
        blue_mask = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
        blue_pixels = np.count_nonzero(blue_mask)
        
        # 洋红色范围（130-170）
        magenta_mask = cv2.inRange(hsv, (130, 50, 100), (170, 255, 255))
        magenta_pixels = np.count_nonzero(magenta_mask)
        
        anomaly_ratio = (blue_pixels + magenta_pixels) / (img.shape[0] * img.shape[1])
    else:
        anomaly_ratio = 0.0
    
    return {
        "laplacian_var": laplacian_var,
        "channel_stds": stds,
        "blue_magenta_ratio": anomaly_ratio
    }

def compare_images(original_path, reconstructed_path, output_json=None):
    """对比两张图像"""
    
    # 读取图像
    orig = cv2.imread(str(original_path))
    recon = cv2.imread(str(reconstructed_path))
    
    if orig is None:
        print(f"❌ 无法读取原始图像: {original_path}")
        return None
    
    if recon is None:
        print(f"❌ 无法读取重构图像: {reconstructed_path}")
        return None
    
    # 调整大小以匹配
    if orig.shape != recon.shape:
        print(f"⚠️ 分辨率不匹配: {orig.shape} vs {recon.shape}")
        h, w = min(orig.shape[0], recon.shape[0]), min(orig.shape[1], recon.shape[1])
        orig = orig[:h, :w]
        recon = recon[:h, :w]
    
    # 计算指标
    metrics = {
        "mse": float(calculate_mse(orig, recon)),
        "psnr": float(calculate_psnr(orig, recon)),
        "ssim": float(calculate_ssim(orig, recon)) if SKIMAGE_AVAILABLE else None,
        "original_analysis": analyze_noise(orig),
        "reconstructed_analysis": analyze_noise(recon),
    }
    
    return metrics

def batch_analyze(results_dir, pattern="*.png"):
    """批量分析目录中的重构图像"""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"❌ 目录不存在: {results_dir}")
        return
    
    images = list(results_dir.glob(pattern))
    if not images:
        print(f"❌ 未找到图像: {results_dir}/{pattern}")
        return
    
    print(f"📊 分析 {len(images)} 张图像...")
    print("=" * 80)
    
    all_metrics = []
    
    for img_path in sorted(images):
        print(f"\n📄 分析: {img_path.name}")
        
        # 尝试找到对应的原始图像
        # 假设命名规则: recovered_images/tag_connX_idxY.png -> original_images/tag_connX_idxY.png
        base_name = img_path.stem
        orig_path = results_dir.parent / "original_images" / f"{base_name}.png"
        
        if orig_path.exists():
            metrics = compare_images(orig_path, img_path)
            if metrics:
                all_metrics.append({
                    "file": img_path.name,
                    "metrics": metrics
                })
                
                print(f"  质量指标:")
                print(f"    PSNR: {metrics['psnr']:.2f} dB (越高越好，>30优秀)")
                print(f"    MSE:  {metrics['mse']:.4f} (越低越好)")
                if metrics['ssim'] is not None:
                    print(f"    SSIM: {metrics['ssim']:.4f} (越接近1越好)")
                
                recon_analysis = metrics['reconstructed_analysis']
                print(f"  重构图像分析:")
                print(f"    焦点度 (Laplacian): {recon_analysis['laplacian_var']:.2f}")
                print(f"    蓝粉噪点比例: {recon_analysis['blue_magenta_ratio']*100:.2f}%")
                
                # 品质评级
                psnr = metrics['psnr']
                if psnr > 35:
                    grade = "优秀 ✓✓✓"
                elif psnr > 30:
                    grade = "很好 ✓✓"
                elif psnr > 25:
                    grade = "良好 ✓"
                else:
                    grade = "可接受 (需改进)"
                print(f"  综合评分: {grade}")
        else:
            print(f"  ⚠️ 未找到原始图像参考")
    
    # 汇总统计
    if all_metrics:
        print("\n" + "=" * 80)
        print("📈 汇总统计:")
        print("=" * 80)
        
        psnrs = [m['metrics']['psnr'] for m in all_metrics if m['metrics']['psnr'] != float('inf')]
        mses = [m['metrics']['mse'] for m in all_metrics]
        ssims = [m['metrics']['ssim'] for m in all_metrics if m['metrics']['ssim'] is not None]
        
        print(f"PSNR: {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
        print(f"MSE:  {np.mean(mses):.4f} ± {np.std(mses):.4f}")
        if ssims:
            print(f"SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
        
        # 保存结果
        if True:  # 总是保存
            output_json = results_dir / "quality_metrics.json"
            with open(output_json, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            print(f"\n✓ 结果已保存: {output_json}")

def main():
    parser = argparse.ArgumentParser(description="语义通信图像质量评估")
    parser.add_argument("--original", type=str, help="原始图像路径")
    parser.add_argument("--reconstructed", type=str, help="重构图像路径")
    parser.add_argument("--batch", type=str, help="批量分析目录（recovered_images）")
    parser.add_argument("--output", type=str, help="输出 JSON 文件路径（可选）")
    
    args = parser.parse_args()
    
    if args.batch:
        batch_analyze(args.batch)
    elif args.original and args.reconstructed:
        metrics = compare_images(args.original, args.reconstructed, args.output)
        if metrics:
            print("\n" + "=" * 80)
            print("质量评估结果")
            print("=" * 80)
            print(f"PSNR: {metrics['psnr']:.2f} dB")
            print(f"MSE:  {metrics['mse']:.6f}")
            if metrics['ssim'] is not None:
                print(f"SSIM: {metrics['ssim']:.4f}")
            
            print(f"\n原始图像分析:")
            orig_analysis = metrics['original_analysis']
            print(f"  焦点度: {orig_analysis['laplacian_var']:.2f}")
            
            print(f"\n重构图像分析:")
            recon_analysis = metrics['reconstructed_analysis']
            print(f"  焦点度: {recon_analysis['laplacian_var']:.2f}")
            print(f"  蓝粉噪点比例: {recon_analysis['blue_magenta_ratio']*100:.2f}%")
            
            # 保存 JSON
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"\n✓ 结果已保存: {args.output}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

