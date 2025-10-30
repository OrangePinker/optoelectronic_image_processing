import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# 简化的图表生成脚本，避免中文字符显示问题
def plot_training_history_simple(csv_path, save_path='training_history_simple.png', title='Training History'):
    """
    生成简化的训练历史图表，使用英文标签避免字体问题
    """
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Pupil Segmentation Model Training History', fontsize=16, fontweight='bold')
    
    epochs = df['epoch']
    
    # 1. 损失曲线
    axes[0, 0].plot(epochs, df['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, df['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Dice系数
    axes[0, 1].plot(epochs, df['train_dice'], 'g-', label='Training Dice', linewidth=2)
    axes[0, 1].plot(epochs, df['val_dice'], 'orange', label='Validation Dice', linewidth=2)
    axes[0, 1].set_title('Dice Coefficient', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. IoU分数
    axes[1, 0].plot(epochs, df['train_iou'], 'purple', label='Training IoU', linewidth=2)
    axes[1, 0].plot(epochs, df['val_iou'], 'brown', label='Validation IoU', linewidth=2)
    axes[1, 0].set_title('IoU Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 过拟合分析
    train_val_loss_diff = df['train_loss'] - df['val_loss']
    train_val_dice_diff = df['val_dice'] - df['train_dice']
    
    axes[1, 1].plot(epochs, train_val_loss_diff, 'red', label='Loss Gap (Train-Val)', linewidth=2)
    axes[1, 1].plot(epochs, train_val_dice_diff, 'blue', label='Dice Gap (Val-Train)', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gap')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 生成训练摘要
    summary_path = save_path.replace('.png', '_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Training Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        # 最佳性能
        best_val_dice_idx = df['val_dice'].idxmax()
        best_val_iou_idx = df['val_iou'].idxmax()
        min_val_loss_idx = df['val_loss'].idxmin()
        
        f.write(f"Best Validation Dice: {df.loc[best_val_dice_idx, 'val_dice']:.4f} (Epoch {df.loc[best_val_dice_idx, 'epoch']})\n")
        f.write(f"Best Validation IoU: {df.loc[best_val_iou_idx, 'val_iou']:.4f} (Epoch {df.loc[best_val_iou_idx, 'epoch']})\n")
        f.write(f"Minimum Validation Loss: {df.loc[min_val_loss_idx, 'val_loss']:.4f} (Epoch {df.loc[min_val_loss_idx, 'epoch']})\n\n")
        
        # 最终性能
        final_epoch = df.iloc[-1]
        f.write(f"Final Performance (Epoch {final_epoch['epoch']}):\n")
        f.write(f"  Training Loss: {final_epoch['train_loss']:.4f}\n")
        f.write(f"  Validation Loss: {final_epoch['val_loss']:.4f}\n")
        f.write(f"  Training Dice: {final_epoch['train_dice']:.4f}\n")
        f.write(f"  Validation Dice: {final_epoch['val_dice']:.4f}\n")
        f.write(f"  Training IoU: {final_epoch['train_iou']:.4f}\n")
        f.write(f"  Validation IoU: {final_epoch['val_iou']:.4f}\n\n")
        
        # 过拟合分析
        final_loss_gap = final_epoch['train_loss'] - final_epoch['val_loss']
        final_dice_gap = final_epoch['val_dice'] - final_epoch['train_dice']
        
        f.write("Overfitting Analysis:\n")
        f.write(f"  Loss Gap (Train-Val): {final_loss_gap:.4f}\n")
        f.write(f"  Dice Gap (Val-Train): {final_dice_gap:.4f}\n")
        
        if abs(final_loss_gap) < 0.05 and abs(final_dice_gap) < 0.02:
            f.write("  Status: Good balance, minimal overfitting\n")
        elif final_loss_gap < -0.1 or final_dice_gap > 0.05:
            f.write("  Status: Potential overfitting detected\n")
        else:
            f.write("  Status: Acceptable performance\n")
    
    print(f"Simple training history visualization saved to {save_path}")
    print(f"Training summary saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate simple training history plots')
    parser.add_argument('--csv_path', required=True, help='Path to training history CSV file')
    parser.add_argument('--save_path', default='training_history_simple.png', help='Path to save the plot')
    parser.add_argument('--title', default='Training History', help='Title for the plot')
    
    args = parser.parse_args()
    
    plot_training_history_simple(args.csv_path, args.save_path, args.title)