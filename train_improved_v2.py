#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的瞳孔分割模型训练脚本 v2
专门针对测试集效果差的问题进行优化
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pupil_segmentation import (
    EyeDataset, dice_coefficient, iou_score, set_seed, train_model, plot_history
)
from simple_unet import SimpleUNet
from torchvision import transforms
import random
import numpy as np

class RandomSelectiveAugmentation:
    """随机选择部分增强效果的自定义变换类"""
    def __init__(self, img_size=256):
        self.img_size = img_size
        # 定义所有可能的增强变换
        self.augmentations = {
            'horizontal_flip': transforms.RandomHorizontalFlip(p=1.0),
            'rotation': transforms.RandomRotation(degrees=8),
            'affine': transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=None, shear=None),
            'resized_crop': transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.95, 1.05)),
            'color_jitter': transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.12),
            'random_erasing': transforms.RandomErasing(p=1.0, scale=(0.02, 0.08), ratio=(0.5, 2.0))
        }
        
    def __call__(self, img):
        # 基础变换（总是应用）
        img = transforms.Resize((self.img_size, self.img_size))(img)
        
        # 随机选择2-4种增强效果
        num_augmentations = random.randint(2, 4)
        selected_augs = random.sample(list(self.augmentations.keys()), num_augmentations)
        
        # 应用选中的增强
        for aug_name in selected_augs:
            if aug_name == 'horizontal_flip' and random.random() < 0.4:
                img = self.augmentations[aug_name](img)
            elif aug_name == 'random_erasing':
                # 随机擦除需要在ToTensor之后应用
                continue
            elif aug_name != 'horizontal_flip':
                img = self.augmentations[aug_name](img)
        
        # 转换为张量和标准化
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        
        # 如果选中了随机擦除，在这里应用
        if 'random_erasing' in selected_augs and random.random() < 0.15:
            img = self.augmentations['random_erasing'](img)
            
        return img

def get_conservative_transforms(img_size=256):
    """
    获取保守的数据转换，使用随机选择的增强策略
    
    Args:
        img_size: 图像大小
    
    Returns:
        train_transform: 训练集转换
        val_transform: 验证集转换
    """
    
    # 训练集转换 - 使用随机选择的增强策略
    train_transform = RandomSelectiveAugmentation(img_size)
    
    # 验证集转换
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_heavy_dropout_model(in_channels=3, num_classes=1, dropout_rate=0.5):
    """
    创建带重度Dropout的模型以防止过拟合
    
    Args:
        in_channels: 输入通道数
        num_classes: 输出类别数
        dropout_rate: Dropout率
    
    Returns:
        model: 带重度Dropout的模型
    """
    class UNetWithHeavyDropout(nn.Module):
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
    
    # 导入必要的组件
    from simple_unet import DoubleConv, Down, Up, OutConv
    
    return UNetWithHeavyDropout(in_channels, num_classes, dropout_rate)

def create_regularized_optimizer_and_scheduler(model, learning_rate=0.00005):
    """
    创建高度正则化的优化器和学习率调度器
    
    Args:
        model: 模型
        learning_rate: 初始学习率
    
    Returns:
        optimizer: 优化器
        scheduler: 学习率调度器
    """
    # 使用AdamW优化器，增强正则化
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=1e-3,  # 进一步增强L2正则化
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 使用CosineAnnealingLR调度器，更平滑的学习率衰减
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,  # 50个epoch的余弦退火
        eta_min=1e-6  # 最小学习率
    )
    
    return optimizer, scheduler

def create_robust_loss():
    """
    创建鲁棒的损失函数
    
    Returns:
        combined_loss: 鲁棒的组合损失函数
    """
    def dice_loss(pred, target, smooth=1e-6):
        """Dice损失函数"""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def focal_loss(pred, target, alpha=0.25, gamma=2.0):
        """Focal损失函数，处理类别不平衡"""
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = alpha * (1 - pt) ** gamma * bce
        return focal.mean()
    
    def tversky_loss(pred, target, alpha=0.3, beta=0.7, smooth=1e-6):
        """Tversky损失函数，更好地处理假阳性和假阴性"""
        pred = torch.sigmoid(pred)
        tp = (pred * target).sum(dim=(2, 3))
        fp = (pred * (1 - target)).sum(dim=(2, 3))
        fn = ((1 - pred) * target).sum(dim=(2, 3))
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return 1 - tversky.mean()
    
    def combined_loss(pred, target):
        """鲁棒的组合损失：Focal + Dice + Tversky"""
        focal = focal_loss(pred, target)
        dice = dice_loss(pred, target)
        tversky = tversky_loss(pred, target)
        return 0.4 * focal + 0.3 * dice + 0.3 * tversky
    
    return combined_loss

def main():
    parser = argparse.ArgumentParser(description='改进的瞳孔分割模型训练 v2')
    parser.add_argument('--data_dir', type=str, default='./附件2-数据集20250923', help='训练数据目录')
    parser.add_argument('--output_dir', type=str, default='./results_final', help='输出目录')
    parser.add_argument('--epochs', type=int, default=60, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=12, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='学习率')
    parser.add_argument('--img_size', type=int, default=256, help='图像大小')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据转换（保守的数据增强）
    train_transform, val_transform = get_conservative_transforms(args.img_size)
    
    # 数据集
    train_dataset = EyeDataset(
        args.data_dir, 
        transform=train_transform, 
        is_train=True, 
        val_split=args.val_split
    )
    
    val_dataset = EyeDataset(
        args.data_dir, 
        transform=val_transform, 
        is_train=False, 
        val_split=args.val_split
    )
    
    print(f'训练集大小: {len(train_dataset)}')
    print(f'验证集大小: {len(val_dataset)}')
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,  # 减少num_workers避免内存问题
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 模型（带重度Dropout防止过拟合）
    model = create_heavy_dropout_model(
        in_channels=3, 
        num_classes=1, 
        dropout_rate=args.dropout_rate
    ).to(device)
    
    # 优化器和调度器
    optimizer, scheduler = create_regularized_optimizer_and_scheduler(model, args.learning_rate)
    
    # 损失函数
    criterion = create_robust_loss()
    
    print('开始训练瞳孔分割模型...')
    print(f'改进策略:')
    print(f'  - 降低学习率: {args.learning_rate} (进一步降低)')
    print(f'  - 增强正则化: weight_decay=1e-3 (进一步增强)')
    print(f'  - 减小批次大小: {args.batch_size} (从16降至12)')
    print(f'  - 增加早停耐心: {args.early_stopping_patience} (从8增至10)')
    print(f'  - 增加重度Dropout: {args.dropout_rate} (从0.3增至0.5)')
    print(f'  - 增加验证集比例: {args.val_split} (从0.15增至0.2)')
    print(f'  - 保守数据增强: 减少增强强度，scale=(0.7, 1.0)')
    print(f'  - 使用Focal+Dice+Tversky损失')
    print(f'  - 使用CosineAnnealingLR调度器')
    
    print(f'训练参数:')
    print(f'  - 训练轮数: {args.epochs}')
    print(f'  - 批次大小: {args.batch_size}')
    print(f'  - 学习率: {args.learning_rate}')
    print(f'  - 图像大小: {args.img_size}')
    print(f'  - 早停耐心值: {args.early_stopping_patience}')
    
    # 训练模型
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        scheduler=scheduler,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # 保存训练历史
    import csv
    history_path = os.path.join(args.output_dir, 'training_history.csv')
    with open(history_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_dice', 'val_dice', 'train_iou', 'val_iou'])
        for i in range(len(history['train_loss'])):
            writer.writerow([
                i+1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_dice'][i],
                history['val_dice'][i],
                history['train_iou'][i],
                history['val_iou'][i]
            ])

    # 绘制训练历史
    plot_history(history, os.path.join(args.output_dir, 'training_history.png'))
    
    print('训练完成!')
    print(f'最佳验证Dice: {max(history["val_dice"]):.4f}')
    print(f'最佳验证IoU: {max(history["val_iou"]):.4f}')
    print(f'结果保存在: {args.output_dir}')
    
    # 计算训练验证差距
    final_train_dice = history['train_dice'][-1]
    final_val_dice = history['val_dice'][-1]
    overfitting_gap = final_train_dice - final_val_dice
    print(f'最终训练-验证Dice差距: {overfitting_gap:.4f}')
    
    if overfitting_gap < 0.03:
        print('✅ 过拟合控制优秀!')
    elif overfitting_gap < 0.05:
        print('✅ 过拟合控制良好!')
    elif overfitting_gap < 0.1:
        print('⚠️ 轻微过拟合')
    else:
        print('❌ 存在过拟合问题')

if __name__ == '__main__':
    main()