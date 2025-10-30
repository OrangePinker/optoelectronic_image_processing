#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进模型v2专用测试脚本
确保模型结构与训练时完全一致
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import time
import argparse
from pupil_segmentation import dice_coefficient, iou_score, set_seed
from simple_unet import DoubleConv, Down, Up, OutConv

class UNetWithHeavyDropout(nn.Module):
    """带重度Dropout的UNet模型，与训练时保持一致"""
    def __init__(self, in_channels, num_classes, dropout_rate=0.5):
        super().__init__()
        base_c = 64
        
        # 编码器部分
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16)
        
        # Dropout层
        self.dropout2d = nn.Dropout2d(dropout_rate)
        self.dropout2d_light = nn.Dropout2d(dropout_rate * 0.5)
        
        # 解码器部分
        self.up1 = Up(base_c * 16, base_c * 8)
        self.up2 = Up(base_c * 8, base_c * 4)
        self.up3 = Up(base_c * 4, base_c * 2)
        self.up4 = Up(base_c * 2, base_c)
        
        # 输出层
        self.out_conv = OutConv(base_c, num_classes)
        
    def forward(self, x):
        # 编码器路径
        x1 = self.in_conv(x)
        x1 = self.dropout2d_light(x1)  # 轻度dropout
        
        x2 = self.down1(x1)
        x2 = self.dropout2d_light(x2)
        
        x3 = self.down2(x2)
        x3 = self.dropout2d_light(x3)
        
        x4 = self.down3(x3)
        x4 = self.dropout2d(x4)  # 重度dropout
        
        x5 = self.down4(x4)
        x5 = self.dropout2d(x5)  # 重度dropout在瓶颈层
        
        # 解码器路径
        x = self.up1(x5, x4)
        x = self.dropout2d_light(x)
        
        x = self.up2(x, x3)
        x = self.dropout2d_light(x)
        
        x = self.up3(x, x2)
        x = self.dropout2d_light(x)
        
        x = self.up4(x, x1)
        
        return self.out_conv(x)

class TestDataset(Dataset):
    """测试数据集"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 获取所有图像文件
        self.image_files = []
        self.mask_files = []
        
        # 从Original_images文件夹读取图像
        image_dir = os.path.join(data_dir, 'Original_images')
        gt_dir = os.path.join(data_dir, 'Ground_Truth')
        
        if os.path.exists(image_dir):
            for file in os.listdir(image_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(image_dir, file)
                    self.image_files.append(image_path)
                    
                    # 查找对应的ground truth文件
                    base_name = os.path.splitext(file)[0]
                    for gt_file in os.listdir(gt_dir):
                        if gt_file.startswith(base_name) and gt_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            gt_path = os.path.join(gt_dir, gt_file)
                            self.mask_files.append(gt_path)
                            break
                    else:
                        # 如果没有找到对应的GT文件，添加None
                        self.mask_files.append(None)
        
        print(f"找到 {len(self.image_files)} 张测试图像")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 加载mask（如果存在）
        mask = None
        if self.mask_files[idx] is not None:
            mask = Image.open(self.mask_files[idx]).convert('L')
            if self.transform:
                mask = transforms.Resize((256, 256))(mask)
                mask = transforms.ToTensor()(mask)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, mask, os.path.basename(image_path)

def test_improved_model_v2(model_path, test_dir, output_dir, device):
    """
    测试改进后的模型v2
    
    Args:
        model_path: 模型权重路径
        test_dir: 测试数据目录
        output_dir: 输出目录
        device: 计算设备
    
    Returns:
        metrics: 性能指标字典
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集和数据加载器
    test_dataset = TestDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 创建模型（与训练时结构完全一致）
    model = UNetWithHeavyDropout(in_channels=3, num_classes=1, dropout_rate=0.5).to(device)
    
    # 加载模型权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ 成功加载模型权重: {model_path}")
    except Exception as e:
        print(f"❌ 加载模型权重失败: {e}")
        raise RuntimeError(f"无法加载模型权重: {e}")
    
    model.eval()
    
    # 性能指标
    dice_scores = []
    iou_scores = []
    processing_times = []
    
    print(f"开始测试，共 {len(test_dataset)} 张图像...")
    
    with torch.no_grad():
        for i, (image, mask, filename) in enumerate(test_loader):
            image = image.to(device)
            
            # 记录处理时间
            start_time = time.time()
            
            # 模型推理
            output = model(image)
            output = torch.sigmoid(output)
            
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            # 转换为numpy数组
            pred_mask = output.cpu().numpy()[0, 0]
            pred_binary = (pred_mask > 0.5).astype(np.uint8)
            
            # 原始图像（用于可视化）
            original_image = image.cpu().numpy()[0]
            original_image = np.transpose(original_image, (1, 2, 0))
            # 反归一化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            original_image = original_image * std + mean
            original_image = np.clip(original_image, 0, 1)
            original_image = (original_image * 255).astype(np.uint8)
            
            # 计算性能指标（如果有ground truth）
            if mask is not None:
                gt_mask = mask.cpu().numpy()[0, 0]
                gt_binary = (gt_mask > 0.5).astype(np.uint8)
                
                dice = dice_coefficient(torch.tensor(pred_binary), torch.tensor(gt_binary))
                iou = iou_score(torch.tensor(pred_binary), torch.tensor(gt_binary))
                
                dice_scores.append(dice.item())
                iou_scores.append(iou.item())
            
            # 保存可视化结果
            base_name = os.path.splitext(filename[0])[0]
            
            # 1. 保存原始预测结果
            result_path = os.path.join(output_dir, 'visualizations', f'result_{base_name}.png')
            pred_vis = (pred_mask * 255).astype(np.uint8)
            cv2.imwrite(result_path, pred_vis)
            
            # 2. 保存分割图像（保留瞳孔区域，其他区域为黑色）
            segmented_image = original_image.copy()
            mask_3channel = np.stack([pred_binary, pred_binary, pred_binary], axis=2)
            segmented_image = segmented_image * mask_3channel
            
            segmented_path = os.path.join(output_dir, 'visualizations', f'segmented_{base_name}.png')
            cv2.imwrite(segmented_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
            print(f"保存分割图像: segmented_{base_name}.png")
            
            # 3. 保存椭圆轮廓图像
            contour_image = original_image.copy()
            
            # 查找轮廓
            contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 拟合椭圆（需要至少5个点）
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    # 在原图上绘制绿色椭圆
                    cv2.ellipse(contour_image, ellipse, (0, 255, 0), 2)
            
            contour_path = os.path.join(output_dir, 'visualizations', f'contour_{base_name}.png')
            cv2.imwrite(contour_path, cv2.cvtColor(contour_image, cv2.COLOR_RGB2BGR))
            print(f"保存轮廓图像: contour_{base_name}.png")
            
            print(f"处理进度: {i+1}/{len(test_dataset)}")
    
    # 计算平均指标
    avg_processing_time = np.mean(processing_times)
    fps = 1.0 / avg_processing_time
    
    metrics = {
        'num_images': len(test_dataset),
        'avg_processing_time': avg_processing_time,
        'fps': fps,
        'avg_dice': np.mean(dice_scores) if dice_scores else 0.0,
        'avg_iou': np.mean(iou_scores) if iou_scores else 0.0,
        'dice_scores': dice_scores,
        'iou_scores': iou_scores
    }
    
    return metrics

def save_test_report(metrics, output_dir):
    """保存测试报告"""
    report_path = os.path.join(output_dir, 'test_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("============================================================\n")
        f.write("                改进模型v2测试结果\n")
        f.write("============================================================\n")
        f.write(f"测试图像数量: {metrics['num_images']}\n")
        f.write(f"平均处理时间: {metrics['avg_processing_time']:.4f} 秒/图像\n")
        f.write(f"处理速度: {metrics['fps']:.2f} FPS\n\n")
        
        if metrics['dice_scores']:
            f.write("分割性能指标:\n")
            f.write(f"  - 平均 Dice 系数: {metrics['avg_dice']:.4f}\n")
            f.write(f"  - 平均 IoU 分数: {metrics['avg_iou']:.4f}\n\n")
            
            f.write("详细指标:\n")
            for i, (dice, iou) in enumerate(zip(metrics['dice_scores'], metrics['iou_scores'])):
                f.write(f"  图像 {i:03d}: Dice={dice:.4f}, IoU={iou:.4f}\n")
        else:
            f.write("注意: 未找到Ground Truth文件，无法计算分割性能指标\n")
        
        f.write("============================================================\n")

def main():
    parser = argparse.ArgumentParser(description='测试改进模型v2')
    parser.add_argument('--model_path', type=str, 
                       default='./results_final/best_model.pth',
                       help='模型权重路径')
    parser.add_argument('--test_dir', type=str, 
                       default='./附件1-测试',
                       help='测试数据目录')
    parser.add_argument('--output_dir', type=str, 
                       default='./test_results_final',
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 测试模型
        metrics = test_improved_model_v2(
            model_path=args.model_path,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            device=device
        )
        
        # 打印结果
        print("\n" + "="*60)
        print("                改进模型v2测试结果")
        print("="*60)
        print(f"测试图像数量: {metrics['num_images']}")
        print(f"平均处理时间: {metrics['avg_processing_time']:.4f} 秒/图像")
        print(f"处理速度: {metrics['fps']:.2f} FPS")
        print()
        
        if metrics['dice_scores']:
            print("分割性能指标:")
            print(f"  - 平均 Dice 系数: {metrics['avg_dice']:.4f}")
            print(f"  - 平均 IoU 分数: {metrics['avg_iou']:.4f}")
        else:
            print("注意: 未找到Ground Truth文件，无法计算分割性能指标")
        
        print("="*60)
        print()
        
        # 保存报告
        save_test_report(metrics, args.output_dir)
        print(f"详细报告已保存至: {args.output_dir}\\test_report.txt")
        print(f"可视化结果已保存至: {args.output_dir}\\visualizations")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        raise

if __name__ == '__main__':
    main()