# 瞳孔分割项目 - 使用说明

## 项目概述

本项目是一个基于深度学习的瞳孔分割系统，使用改进的U-Net模型对眼部图像进行瞳孔区域的精确分割。项目包含完整的训练、测试和可视化功能。

## 核心程序文件

### 1. 训练程序

#### `train_improved_v2.py` - 主要训练脚本
**功能**: 训练改进的瞳孔分割模型，包含多种优化策略

**主要特性**:
- 带重度Dropout的U-Net模型
- 保守的数据增强策略
- Focal+Dice+Tversky复合损失函数
- 余弦退火学习率调度
- 早停机制防止过拟合

**使用方法**:
```bash
# 基本使用
python train_improved_v2.py

# 自定义参数
python train_improved_v2.py --epochs 60 --batch_size 12 --learning_rate 0.00005 --dropout_rate 0.5
```

**参数说明**:
- `--data_dir`: 训练数据目录 (默认: `./附件2-数据集20250923`)
- `--output_dir`: 输出目录 (默认: `./results_final`)
- `--epochs`: 训练轮数 (默认: 60)
- `--batch_size`: 批次大小 (默认: 12)
- `--learning_rate`: 学习率 (默认: 0.00005)
- `--img_size`: 图像大小 (默认: 256)
- `--val_split`: 验证集比例 (默认: 0.2)
- `--early_stopping_patience`: 早停耐心值 (默认: 10)
- `--dropout_rate`: Dropout率 (默认: 0.5)
- `--seed`: 随机种子 (默认: 123)

**输出文件**:
- `best_model.pth`: 最佳模型权重
- `training_history.csv`: 训练历史数据
- `training_history.png`: 训练过程可视化

### 2. 测试程序

#### `test_improved_v2.py` - 主要测试脚本
**功能**: 测试训练好的模型，生成分割结果和性能报告

**主要特性**:
- 与训练时完全一致的模型结构
- 自动计算Dice系数和IoU分数
- 生成详细的可视化结果
- 性能分析和报告生成

**使用方法**:
```bash
# 基本使用
python test_improved_v2.py

# 自定义参数
python test_improved_v2.py --model_path ./results_final/best_model.pth --output_dir ./test_results_final
```

**参数说明**:
- `--model_path`: 模型权重路径 (默认: `./results_final/best_model.pth`)
- `--test_dir`: 测试数据目录 (默认: `./附件1-测试`)
- `--output_dir`: 输出目录 (默认: `./test_results_final`)
- `--seed`: 随机种子 (默认: 42)

**输出文件**:
- `test_report.txt`: 详细测试报告
- `performance_report.txt`: 性能分析报告
- `visualizations/`: 可视化结果目录
  - `result_*.png`: 分割结果对比图
  - `segmented_*.png`: 分割掩码图
  - `contour_*.png`: 轮廓检测图

### 3. 辅助程序

#### `enhanced_plot_history_simple.py` - 训练历史可视化
**功能**: 生成增强版的训练历史图表和分析报告

**使用方法**:
```bash
python enhanced_plot_history_simple.py --csv_path ./results_final/training_history.csv
```

**参数说明**:
- `--csv_path`: 训练历史CSV文件路径 (必需)
- `--save_path`: 图表保存路径 (默认: `training_history_simple.png`)
- `--title`: 图表标题 (默认: `Training History`)

**输出文件**:
- `training_history_simple.png`: 训练历史图表
- `training_history_simple_summary.txt`: 训练摘要报告

#### `pupil_segmentation.py` - 核心功能模块
**功能**: 提供数据集处理、模型训练、评估等核心功能

**主要组件**:
- `EyeDataset`: 眼部图像数据集类
- `dice_coefficient`: Dice系数计算
- `iou_score`: IoU分数计算
- `train_model`: 模型训练函数
- `plot_history`: 训练历史可视化
- `test_model`: 模型测试函数

#### `simple_unet.py` - U-Net模型定义
**功能**: 定义简化的U-Net模型结构

**主要组件**:
- `DoubleConv`: 双卷积块
- `Down`: 下采样块
- `Up`: 上采样块
- `OutConv`: 输出卷积层
- `SimpleUNet`: 完整的U-Net模型

## 完整使用流程

### 1. 环境准备
```bash
# 确保安装必要的依赖包
pip install torch torchvision opencv-python pillow numpy pandas matplotlib tqdm
```

### 2. 数据准备
- 训练数据: 放置在 `./附件2-数据集20250923/` 目录
- 测试数据: 放置在 `./附件1-测试/` 目录

### 3. 模型训练
```bash
# 使用推荐参数训练模型
python train_improved_v2.py --epochs 60 --batch_size 12 --learning_rate 0.00005 --dropout_rate 0.5
```

### 4. 生成训练历史图表
```bash
# 生成增强版训练历史可视化
python enhanced_plot_history_simple.py --csv_path ./results_final/training_history.csv
```

### 5. 模型测试
```bash
# 测试训练好的模型
python test_improved_v2.py --model_path ./results_final/best_model.pth --output_dir ./test_results_final
```

## 项目结构

```
optoelectronic_image_processing/
├── train_improved_v2.py          # 主要训练脚本
├── test_improved_v2.py           # 主要测试脚本
├── enhanced_plot_history_simple.py  # 训练历史可视化
├── pupil_segmentation.py         # 核心功能模块
├── simple_unet.py               # U-Net模型定义
├── 附件2-数据集20250923/         # 训练数据目录
├── 附件1-测试/                  # 测试数据目录
├── results_final/               # 训练结果输出
│   ├── best_model.pth          # 最佳模型权重
│   ├── training_history.csv    # 训练历史数据
│   └── training_history.png    # 训练历史图表
├── test_results_final/          # 测试结果输出
│   ├── test_report.txt         # 测试报告
│   ├── performance_report.txt  # 性能报告
│   └── visualizations/         # 可视化结果
└── README.md                   # 项目说明文档
```

## 性能指标

最新模型性能表现:
- **验证集Dice系数**: 0.9673
- **验证集IoU分数**: 0.9367
- **测试集平均Dice**: 0.9187
- **测试集平均IoU**: 0.8513
- **处理速度**: 49.81 FPS

## 模型特点

1. **改进的U-Net架构**: 带重度Dropout防止过拟合
2. **保守数据增强**: scale=(0.7, 1.0)，避免过度变形
3. **复合损失函数**: Focal+Dice+Tversky损失
4. **学习率调度**: 余弦退火调度器
5. **早停机制**: 防止过拟合，耐心值为10

## 注意事项

1. **GPU推荐**: 建议使用GPU进行训练以提高速度
2. **内存要求**: 批次大小12需要约8GB显存
3. **数据格式**: 支持常见图像格式(PNG, JPG等)
4. **随机种子**: 已设置固定种子确保结果可复现

## 故障排除

1. **CUDA内存不足**: 减小batch_size参数
2. **训练速度慢**: 检查是否正确使用GPU
3. **模型不收敛**: 尝试调整学习率或增加训练轮数
4. **测试结果差**: 确保测试数据格式与训练数据一致

## 联系信息

如有问题或建议，请查看项目文档或联系开发团队。