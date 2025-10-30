import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from tqdm import tqdm
import sys
import time
import argparse
import csv

# 移除matplotlib相关代码，完全使用OpenCV替代
# 使用OpenCV进行可视化（移除print语句避免重复输出）

# 添加U-Net模型路径
sys.path.append(os.path.join(os.path.dirname(__file__), '模型参考U-Net', 'U-Net'))
try:
    from src.model import UNet
except ImportError:
    # 如果导入失败，尝试直接从模型文件导入
    import importlib.util
    model_path = os.path.join(os.path.dirname(__file__), '模型参考U-Net', 'U-Net', 'src', 'model.py')
    spec = importlib.util.spec_from_file_location("model", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    UNet = model_module.UNet

# 导入简化的UNet模型作为备选
from simple_unet import SimpleUNet

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 定义眼科图像数据集类
class EyeDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None, is_train=True, val_split=0.2):
        """
        初始化眼科图像数据集
        
        Args:
            data_dir: 数据集根目录
            transform: 输入图像的转换
            target_transform: 标签图像的转换
            is_train: 是否为训练集
            val_split: 验证集比例
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # 获取所有子文件夹
        self.folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        
        # 收集所有图像对
        self.image_pairs = []
        for folder in self.folders:
            orig_dir = os.path.join(data_dir, folder, 'Original_images')
            label_dir = os.path.join(data_dir, folder, 'Labels')
            
            if os.path.exists(orig_dir) and os.path.exists(label_dir):
                images = [f for f in os.listdir(orig_dir) if f.endswith('.jpg')]
                
                for img_name in images:
                    img_path = os.path.join(orig_dir, img_name)
                    label_path = os.path.join(label_dir, img_name)
                    
                    if os.path.exists(label_path):
                        self.image_pairs.append((img_path, label_path))
        
        # 打乱数据集
        random.shuffle(self.image_pairs)
        
        # 划分训练集和验证集
        val_size = int(len(self.image_pairs) * val_split)
        if is_train:
            self.image_pairs = self.image_pairs[val_size:]
        else:
            self.image_pairs = self.image_pairs[:val_size]
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        img_path, label_path = self.image_pairs[idx]
        
        # 读取图像和标签
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path).convert('L')  # 转换为灰度图
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            # 如果没有提供target_transform，使用默认转换
            # 确保掩码与输入图像尺寸一致
            mask = transforms.Resize((256, 256))(mask)  # 改回256x256
            mask = transforms.ToTensor()(mask)
        
        # 二值化掩码
        mask = (mask > 0.5).float()
        
        return image, mask

# 定义数据转换
def get_transforms(img_size=256):
    """
    获取数据转换（增强版）
    
    Args:
        img_size: 图像大小
    
    Returns:
        train_transform: 训练集转换
        val_transform: 验证集转换
    """
    # 训练集转换 - 增强数据增强策略
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # 增加旋转角度
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 添加仿射变换
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),  # 增强颜色抖动
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),  # 添加高斯模糊
        transforms.RandomApply([transforms.ElasticTransform(alpha=50.0, sigma=5.0)], p=0.3),  # 添加弹性变换
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))  # 添加随机擦除
    ])
    
    # 验证集转换
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# 计算Dice系数
def dice_coefficient(pred, target):
    """
    计算Dice系数
    
    Args:
        pred: 预测掩码
        target: 目标掩码
    
    Returns:
        dice: Dice系数
    """
    smooth = 1e-5
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

# 计算IoU
def iou_score(pred, target):
    """
    计算IoU分数
    
    Args:
        pred: 预测掩码
        target: 目标掩码
    
    Returns:
        iou: IoU分数
    """
    smooth = 1e-5
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, output_dir='./results', 
                scheduler=None, early_stopping_patience=10):
    """
    训练模型（增强版）
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        num_epochs: 训练轮数
        output_dir: 输出目录
        scheduler: 学习率调度器
        early_stopping_patience: 早停耐心值
    
    Returns:
        history: 训练历史
    """
    save_path = os.path.join(output_dir, 'best_model.pth')
    best_val_dice = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': [], 'train_iou': [], 'val_iou': []}
    
    # 早停机制
    patience_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_iou = 0.0
        
        # 记录训练时间
        train_start_time = time.time()
        
        for inputs, masks in tqdm(train_loader):
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * inputs.size(0)
            
            # 计算Dice系数和IoU
            preds = torch.sigmoid(outputs) > 0.5
            train_dice += dice_coefficient(preds.float(), masks).item() * inputs.size(0)
            train_iou += iou_score(preds.float(), masks).item() * inputs.size(0)
        
        train_time = time.time() - train_start_time
        
        # 计算平均损失和指标
        train_loss = train_loss / len(train_loader.dataset)
        train_dice = train_dice / len(train_loader.dataset)
        train_iou = train_iou / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        
        # 记录验证时间
        val_start_time = time.time()
        
        with torch.no_grad():
            for inputs, masks in tqdm(val_loader):
                inputs = inputs.to(device)
                masks = masks.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                
                # 统计
                val_loss += loss.item() * inputs.size(0)
                
                # 计算Dice系数和IoU
                preds = torch.sigmoid(outputs) > 0.5
                val_dice += dice_coefficient(preds.float(), masks).item() * inputs.size(0)
                val_iou += iou_score(preds.float(), masks).item() * inputs.size(0)
        
        val_time = time.time() - val_start_time
        
        # 计算平均损失和指标
        val_loss = val_loss / len(val_loader.dataset)
        val_dice = val_dice / len(val_loader.dataset)
        val_iou = val_iou / len(val_loader.dataset)
        
        # 学习率调度
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # 保存最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f'Saved best model with Dice: {val_dice:.4f}')
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        
        # 显示当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f}, Train Time: {train_time:.2f}s')
        print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Val Time: {val_time:.2f}s')
        print(f'Learning Rate: {current_lr:.6f}, Patience: {patience_counter}/{early_stopping_patience}')
    
    return history

# 绘制训练历史
def plot_history(history, save_path='training_history.png'):
    """
    可视化训练历史
    
    Args:
        history: 训练历史
        save_path: 保存路径
    """
    # 保存训练历史数据到CSV文件
    csv_path = save_path.replace('.png', '.csv') if save_path else 'training_history.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_dice', 'val_dice', 'train_iou', 'val_iou'])
        for i in range(len(history['train_loss'])):
            writer.writerow([i+1, history['train_loss'][i], history['val_loss'][i], 
                            history['train_dice'][i], history['val_dice'][i],
                            history['train_iou'][i], history['val_iou'][i]])
    print(f"训练历史数据已保存到 {csv_path}")
    
    # 使用OpenCV创建可视化图像
    h, w = 400, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # 绘制网格和边框
    cv2.line(img, (w//2, 0), (w//2, h), (200, 200, 200), 1)
    cv2.rectangle(img, (0, 0), (w-1, h-1), (0, 0, 0), 1)
    
    # 获取数据范围
    epochs = len(history['train_loss'])
    max_loss = max(max(history['train_loss']), max(history['val_loss'])) * 1.1
    max_metric = max(max(history['train_dice']), max(history['val_dice']), 
                     max(history['train_iou']), max(history['val_iou'])) * 1.1
    
    # 绘制损失曲线
    for i in range(1, epochs):
        # 训练损失
        pt1 = (int((i-1) * (w//2) / epochs), int(h - history['train_loss'][i-1] * h / max_loss))
        pt2 = (int(i * (w//2) / epochs), int(h - history['train_loss'][i] * h / max_loss))
        cv2.line(img, pt1, pt2, (255, 0, 0), 2)
        
        # 验证损失
        pt1 = (int((i-1) * (w//2) / epochs), int(h - history['val_loss'][i-1] * h / max_loss))
        pt2 = (int(i * (w//2) / epochs), int(h - history['val_loss'][i] * h / max_loss))
        cv2.line(img, pt1, pt2, (0, 0, 255), 2)
    
    # 绘制指标曲线
    for i in range(1, epochs):
        # 训练Dice
        pt1 = (int((i-1) * (w//2) / epochs) + w//2, int(h - history['train_dice'][i-1] * h / max_metric))
        pt2 = (int(i * (w//2) / epochs) + w//2, int(h - history['train_dice'][i] * h / max_metric))
        cv2.line(img, pt1, pt2, (255, 0, 0), 2)
        
        # 验证Dice
        pt1 = (int((i-1) * (w//2) / epochs) + w//2, int(h - history['val_dice'][i-1] * h / max_metric))
        pt2 = (int(i * (w//2) / epochs) + w//2, int(h - history['val_dice'][i] * h / max_metric))
        cv2.line(img, pt1, pt2, (0, 0, 255), 2)
        
        # 训练IoU
        pt1 = (int((i-1) * (w//2) / epochs) + w//2, int(h - history['train_iou'][i-1] * h / max_metric))
        pt2 = (int(i * (w//2) / epochs) + w//2, int(h - history['train_iou'][i] * h / max_metric))
        cv2.line(img, pt1, pt2, (0, 255, 0), 2)
        
        # 验证IoU
        pt1 = (int((i-1) * (w//2) / epochs) + w//2, int(h - history['val_iou'][i-1] * h / max_metric))
        pt2 = (int(i * (w//2) / epochs) + w//2, int(h - history['val_iou'][i] * h / max_metric))
        cv2.line(img, pt1, pt2, (255, 0, 255), 2)
    
    # 添加标题和图例
    cv2.putText(img, "Training Loss", (w//4-40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "Training Metrics", (3*w//4-50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # 添加图例
    cv2.putText(img, "Train Loss", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.putText(img, "Val Loss", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(img, "Train Dice", (w//2+10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.putText(img, "Val Dice", (w//2+10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(img, "Train IoU", (w//2+10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(img, "Val IoU", (w//2+10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    
    cv2.imwrite(save_path, img)
    print(f"训练历史可视化已保存到 {save_path}")

# 测试模型
def test_model(model, test_loader, device, output_dir='./results'):
    """
    测试模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        output_dir: 输出目录
    
    Returns:
        metrics: 评估指标
    """
    model.eval()
    dice_scores = []
    iou_scores = []
    contour_overlaps = []
    processing_times = []
    
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    with torch.no_grad():
        for i, (inputs, masks) in enumerate(tqdm(test_loader)):
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            # 记录处理时间
            start_time = time.time()
            
            # 前向传播
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            
            # 计算处理时间
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # 计算Dice系数和IoU
            batch_dice = dice_coefficient(preds.float(), masks).item()
            batch_iou = iou_score(preds.float(), masks).item()
            
            dice_scores.append(batch_dice)
            iou_scores.append(batch_iou)
            
            # 转换为NumPy数组
            image_np = inputs.squeeze().cpu().numpy().transpose(1, 2, 0)
            mask_np = masks.squeeze().cpu().numpy()
            pred_np = preds.squeeze().cpu().numpy()
            
            # 反归一化图像
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image_np * std + mean
            image_np = np.clip(image_np, 0, 1)
            
            # 拟合瞳孔轮廓
            gt_contour, gt_ellipse = fit_pupil_contour(mask_np)
            pred_contour, pred_ellipse = fit_pupil_contour(pred_np)
            
            # 计算轮廓重叠度
            if gt_contour is not None and pred_contour is not None:
                overlap = contour_overlap(pred_contour, gt_contour, mask_np.shape)
                contour_overlaps.append(overlap)
            
            # 可视化结果 - 原始格式
            save_path = os.path.join(output_dir, 'visualizations', f'result_{i}.png')
            visualize_results(image_np, mask_np, pred_np, pred_contour, pred_ellipse, save_path)
            
            # 可视化结果 - 用户自定义格式
            custom_dir = os.path.join(output_dir, 'custom_visualizations')
            visualize_custom_results(image_np, pred_np, pred_contour, pred_ellipse, custom_dir, i)
    
    # 计算平均指标
    metrics = {
        'dice': np.mean(dice_scores),
        'iou': np.mean(iou_scores),
        'contour_overlap': np.mean(contour_overlaps) if contour_overlaps else 0,
        'time_per_image': np.mean(processing_times)
    }
    
    return metrics

# 瞳孔区外轮廓拟合
def fit_pupil_contour(mask):
    """
    拟合瞳孔区外轮廓
    
    Args:
        mask: 二值掩码
    
    Returns:
        contour: 拟合的轮廓
        ellipse: 拟合的椭圆参数
    """
    # 转换为uint8类型
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # 找到最大轮廓
    max_contour = max(contours, key=cv2.contourArea)
    
    # 拟合椭圆
    if len(max_contour) >= 5:  # 至少需要5个点才能拟合椭圆
        ellipse = cv2.fitEllipse(max_contour)
        return max_contour, ellipse
    else:
        return max_contour, None

# 计算轮廓重叠度
def contour_overlap(pred_contour, gt_contour, image_shape):
    """
    计算轮廓重叠度
    
    Args:
        pred_contour: 预测轮廓
        gt_contour: 真实轮廓
        image_shape: 图像形状
    
    Returns:
        overlap: 重叠度
    """
    if pred_contour is None or gt_contour is None:
        return 0.0
    
    # 创建掩码
    pred_mask = np.zeros(image_shape, dtype=np.uint8)
    gt_mask = np.zeros(image_shape, dtype=np.uint8)
    
    # 绘制轮廓
    cv2.drawContours(pred_mask, [pred_contour], 0, 1, -1)
    cv2.drawContours(gt_mask, [gt_contour], 0, 1, -1)
    
    # 计算重叠度
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    return intersection / union if union > 0 else 0.0

# 可视化分割结果
def visualize_results(image, mask, pred, contour=None, ellipse=None, save_path=None):
    """
    可视化分割结果
    
    Args:
        image: 原始图像
        mask: 真实掩码
        pred: 预测掩码
        contour: 拟合的轮廓
        ellipse: 拟合的椭圆参数
        save_path: 保存路径
    """
    # 创建可视化结果并直接保存为图像
    h, w = mask.shape[:2]
    result_img = np.zeros((h*2, w*3, 3), dtype=np.uint8)
    
    # 原始图像
    if len(image.shape) == 3:
        orig_img = (image * 255).astype(np.uint8)
    else:
        orig_img = np.repeat(image[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8)
    
    # 真实掩码
    gt_mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
    gt_mask_vis[mask > 0] = [0, 255, 0]
    
    # 预测掩码
    pred_mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
    pred_mask_vis[pred > 0] = [0, 255, 0]
    
    # 原始图像 + 真实掩码
    gt_overlay = orig_img.copy()
    gt_overlay[mask > 0] = [0, 255, 0]
    
    # 原始图像 + 预测掩码
    pred_overlay = orig_img.copy()
    pred_overlay[pred > 0] = [0, 255, 0]
    
    # 原始图像 + 预测轮廓 + 椭圆拟合
    contour_overlay = orig_img.copy()
    if contour is not None:
        cv2.drawContours(contour_overlay, [contour], 0, (0, 255, 0), 2)
    
    # 绘制椭圆拟合
    if ellipse is not None:
        cv2.ellipse(contour_overlay, ellipse, (0, 255, 0), 2)
    
    # 组合结果
    result_img[0:h, 0:w] = orig_img
    result_img[0:h, w:2*w] = gt_mask_vis
    result_img[0:h, 2*w:3*w] = pred_mask_vis
    result_img[h:2*h, 0:w] = gt_overlay
    result_img[h:2*h, w:2*w] = pred_overlay
    result_img[h:2*h, 2*w:3*w] = contour_overlay
    
    # 添加标题
    cv2.putText(result_img, "Original", (w//2-40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(result_img, "GT Mask", (w+w//2-40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(result_img, "Pred Mask", (2*w+w//2-40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(result_img, "GT Overlay", (w//2-40, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(result_img, "Pred Overlay", (w+w//2-40, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(result_img, "Contour+Ellipse", (2*w+w//2-60, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 保存结果
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, result_img)
    
    return result_img


def visualize_custom_results(image, pred, contour=None, ellipse=None, save_dir=None, image_idx=0):
    """
    根据用户要求生成自定义可视化结果
    1. 输出分割后的图像（除目标瞳孔图像外，其他标黑，不另外加颜色）
    2. 外轮廓拟合线，即在原图形基础上多加一个绿色椭圆线
    
    Args:
        image: 原始图像
        pred: 预测掩码
        contour: 拟合的轮廓
        ellipse: 拟合的椭圆参数
        save_dir: 保存目录
        image_idx: 图像索引
    """
    # 原始图像处理
    if len(image.shape) == 3:
        orig_img = (image * 255).astype(np.uint8)
    else:
        orig_img = np.repeat(image[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8)
    
    # 1. 生成分割后的图像（瞳孔区域保持原色，其他区域标黑）
    segmented_img = np.zeros_like(orig_img)
    # 只在预测掩码为正的区域保留原图像
    segmented_img[pred > 0] = orig_img[pred > 0]
    
    # 2. 生成带绿色椭圆拟合线的原图
    ellipse_img = orig_img.copy()
    if ellipse is not None:
        # 绘制绿色椭圆拟合线
        cv2.ellipse(ellipse_img, ellipse, (0, 255, 0), 2)
    
    # 保存结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存分割后的图像
        segmented_path = os.path.join(save_dir, f'segmented_{image_idx}.png')
        cv2.imwrite(segmented_path, segmented_img)
        
        # 保存带椭圆拟合线的原图
        ellipse_path = os.path.join(save_dir, f'ellipse_{image_idx}.png')
        cv2.imwrite(ellipse_path, ellipse_img)
        
        return segmented_path, ellipse_path
    
    return segmented_img, ellipse_img