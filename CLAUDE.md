# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**重要提示：请使用中文与用户交流**

## 项目概述

这是一个基于扩散模型的二维材料生成与优化项目，目标是利用MatterGen扩散模型从通用晶体数据库中学习材料结构特征，设计并生成具备高HER催化活性、高稳定性和实验可合成性的新型二维材料。

项目采用迭代优化策略：**材料筛选 → MatterGen训练 → 新材料生成 → 性能评估 → 反馈优化**

## 环境配置

### 依赖安装
```bash
# 创建虚拟环境
conda create -n material_gen python=3.10
conda activate material_gen

# 安装核心依赖
pip install -r requirements.txt

# 单独安装MatterGen（从GitHub）
pip install git+https://github.com/microsoft/mattergen.git

# 安装PyTorch和PyTorch Geometric（根据CUDA版本调整）
pip install torch torchvision torchaudio
pip install torch-geometric
```

## 核心开发命令和工作流程

### 完整材料筛选流程
```bash
# 步骤1：合成性预筛选（元素种类 ≤ 3）
python filter_synthesizable_materials_simple.py
# 输入：data/raw/*.cif
# 输出：data/filtered/*.cif

# 步骤2：二维材料筛选
python run_2d_screening.py
# 输入：data/filtered/*.cif
# 输出：data/2d_materials/*.cif

# 步骤3：形成能筛选（稳定性检查）
python quick_formation_screening.py
# 筛选条件：Formation energy < 0 eV/atom

# 步骤4：转换为2D结构（可选）
python process_formation_to_2d.py
```

### MatterGen数据准备
```bash
# 准备训练数据（从筛选结果）
python prepare_training_data.py

# 准备MatterGen数据格式
python prepare_mattergen_filtered_data.py

# 转换为数据集格式
csv-to-dataset --csv-folder data/mattergen/ --dataset-name 2d_materials --cache-folder data/cache
```

### MatterGen训练

#### 方法1：使用命令行
```bash
# 准备训练数据
csv-to-dataset --csv-folder data/mattergen/ --dataset-name 2d_materials --cache-folder data/cache

# 开始训练 (根据GPU内存调整batch_size)
mattergen-train \
    data_module=2d_materials \
    data_module.batch_size=8 \
    trainer.max_epochs=100 \
    trainer.precision="16-mixed" \
    trainer.devices=1 \
    trainer.log_every_n_steps=50
```

#### 方法2：使用配置文件
```bash
# 使用项目配置文件
mattergen-train --config-path configs --config-name config
```

### 训练监控
```bash
# 启动TensorBoard
tensorboard --logdir logs
# 访问 http://localhost:6006
```

### 材料生成和筛选
```bash
# 生成新材料
python run_mattergen_generation.py
# 或直接使用MatterGen命令
mattergen-generate \
    --checkpoint checkpoints/last.ckpt \
    --num-samples 10 \
    --output-dir generated_materials/ \
    --batch-size 4

# 生成稳定材料（条件生成）
python run_stable_generation.py

# 生成后进行质量筛选
python generate_stable_materials.py
```

### 迭代训练和优化
```bash
# 启动迭代训练流程
python iterative_training.py
# 实现：生成 → 筛选 → 训练 → 改进的循环

# 计算形成能（用于质量评估）
python calculate_formation_energy.py
```

### 开发和调试
```bash
# 启动Jupyter进行交互式开发
jupyter notebook

# 检查数据质量和统计
python analyze_2d_results.py

# TensorBoard监控训练进展
tensorboard --logdir logs
```

## 项目架构

### 目录结构和核心文件
```
项目根目录/
├── data/                           # 数据目录
│   ├── raw/                       # 原始CIF文件
│   ├── filtered/                  # 筛选后材料（≤3元素）
│   ├── 2d_materials/             # 二维材料候选
│   ├── mattergen/                # MatterGen训练数据
│   └── formation_energy_results/ # 形成能计算结果
├── src/                          # 核心源码
│   ├── data/                     # 数据处理模块
│   │   ├── crystal_2d_converter.py      # 2D转换算法
│   │   ├── filter_2d_materials_*.py     # 2D材料筛选
│   │   ├── mattergen_data_converter.py  # MatterGen格式转换
│   │   └── crystal_representation.py    # 晶体特征表示
│   ├── models/                   # 模型相关
│   └── utils/                    # 工具函数
├── configs/                      # 配置文件
│   ├── config.yaml              # 主配置文件
│   ├── 2d_screening_config.yaml # 2D筛选配置
│   └── data_module/             # 数据模块配置
├── checkpoints/                 # 训练检查点
├── generated_materials/         # 生成的材料
└── logs/                       # 训练日志

### 核心脚本功能

#### 数据处理脚本
- `filter_synthesizable_materials_simple.py` - 元素种类筛选（≤3）
- `run_2d_screening.py` - 二维材料识别和筛选
- `quick_formation_screening.py` - 形成能快速筛选
- `prepare_training_data.py` - 准备MatterGen训练数据

#### 训练和生成脚本
- `iterative_training.py` - 迭代训练主控制器
- `run_mattergen_generation.py` - 材料生成脚本
- `run_stable_generation.py` - 稳定材料条件生成
- `calculate_formation_energy.py` - 形成能计算和验证

#### 核心组件
- **TwoDMaterialScreener** (`src/data/filter_2d_materials_*.py`) - 2D材料筛选算法
- **MatterGenDataConverter** (`src/data/mattergen_data_converter.py`) - MatterGen格式转换
- **IterativeTrainer** (`iterative_training.py`) - 迭代训练控制器

## 训练配置参数

### GPU内存配置
- **>= 24GB**: batch_size=16, precision="16-mixed"
- **12-24GB**: batch_size=8, precision="16-mixed" 
- **8-12GB**: batch_size=4, precision="16-mixed"
- **<= 8GB**: batch_size=2, precision=32

### 重要配置
- `configs/config.yaml` - 主配置文件
- `finetune.pretrained_ckpt_path` - 预训练模型路径
- `data.properties_csv` - 属性预测文件
- `model.condition_on_property` - 条件生成开关

## 完整工作流程

### 阶段1：数据筛选和预处理
1. **合成性筛选**：`python filter_synthesizable_materials_simple.py`
   - 筛选≤3元素的材料（提高合成可能性）
   
2. **二维性识别**：`python run_2d_screening.py`  
   - 基于晶体学算法识别层状/二维材料
   
3. **稳定性筛选**：`python quick_formation_screening.py`
   - 形成能 < 0 eV/atom（热力学稳定性）

### 阶段2：MatterGen训练
1. **数据准备**：`python prepare_training_data.py`
   - 转换为MatterGen所需格式
   
2. **模型训练**：`mattergen-train --config-path configs --config-name config`
   - 基于筛选数据训练扩散模型
   
3. **迭代优化**：`python iterative_training.py`
   - 生成 → 评估 → 反馈训练的循环

### 阶段3：材料生成和验证
1. **新材料生成**：`python run_stable_generation.py`
   - 条件生成满足稳定性要求的新材料
   
2. **质量验证**：`python calculate_formation_energy.py`
   - 计算生成材料的形成能等性质
   
3. **结果筛选**：基于计算结果筛选高质量候选材料

## 开发注意事项

### Windows环境特殊设置
- 训练时设置 `num_workers=0`（避免多进程问题）
- 使用反斜杠路径分隔符或`Path`对象处理文件路径
- 大文件处理时注意内存管理

### 性能优化建议
- GPU训练：根据显存选择batch_size（见GPU内存配置）
- 大规模筛选：优先使用快速版本算法
- 长时间训练：配置检查点保存和恢复机制

### 常见问题排查
1. **MatterGen安装问题**：使用`pip install git+https://github.com/microsoft/mattergen.git`
2. **CUDA版本冲突**：单独安装PyTorch和PyTorch Geometric
3. **内存不足**：减少batch_size或使用精度为"16-mixed"
4. **训练中断**：检查checkpoints目录中的保存点

## 技术栈和依赖

- **深度学习框架**: PyTorch, PyTorch Lightning, Hydra
- **材料科学库**: Pymatgen, ASE  
- **图神经网络**: PyTorch Geometric
- **扩散模型**: MatterGen (Microsoft)
- **数据处理**: NumPy, Pandas, scikit-learn
- **可视化**: Matplotlib, Seaborn
- **配置管理**: Hydra, OmegaConf