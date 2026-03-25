# 二维材料筛选系统

基于论文"二维材料的高通量筛选与光催化性能预测"的实现，用于从Materials Project数据库中识别和筛选二维材料。

## 系统概述

该系统实现了两阶段筛选流程：

### 第一阶段：合成性筛选
- **目标**：筛选出元素种类≤3的材料（满足实验合成可能性）
- **脚本**：`filter_synthesizable_materials_simple.py`
- **输入**：`data/raw/` 中的原始CIF文件
- **输出**：`data/filtered/` 中筛选后的材料

### 第二阶段：二维性筛选
- **目标**：从合成性筛选结果中识别真正的二维材料
- **脚本**：`filter_2d_materials_secondary.py`
- **算法**：基于论文中的拓扑缩放算法(TSA)、层状结构分析、结合能估算
- **输入**：`data/filtered/` 中的材料
- **输出**：`data/2d_materials/` 中的二维材料候选

## 核心算法

### 1. 拓扑缩放算法 (TSA)
```python
# 基本原理：通过分析原子团簇在超胞中的缩放关系判断材料维度
scaling_ratio = supercell_clusters / original_clusters

# 维度判断：
# 0D: ratio ≈ 8 (独立分子)
# 1D: ratio ≈ 4 (链状结构)  
# 2D: ratio ≈ 2 (层状结构)
# 3D: ratio ≈ 1 (三维网络)
```

### 2. 层状结构分析
- 层间距离计算
- 晶格各向异性分析
- 化学键网络分析
- 密度分布各向异性

### 3. 结合能估算
```python
# 基于van der Waals相互作用的经验公式
binding_energy = α * contact_area / interlayer_distance^4
```

## 使用方法

### 环境准备
```bash
pip install pymatgen pandas matplotlib seaborn numpy
```

### 运行流程

1. **第一阶段筛选**（如果还没运行过）：
```bash
python filter_synthesizable_materials_simple.py
```

2. **第二阶段筛选** - 有两种版本可选：

**推荐：快速版本**（处理速度 ~100 文件/秒）：
```bash
python run_fast_2d_screening.py
```

**完整版本**（更详细分析但速度较慢）：
```bash
python run_2d_screening.py
```

3. **结果分析**：
```bash
python analyze_2d_results.py
```

### 测试运行
```bash
# 测试快速版本（推荐）
python test_fast_2d_screening.py

# 测试完整版本
python test_2d_screening.py
```

## 输出文件说明

### 筛选结果文件

**快速版本输出**：
- `fast_2d_screening_results.json`：完整筛选结果
- `fast_2d_candidates.json`：二维材料候选
- `fast_2d_summary.csv`：筛选结果摘要表

**完整版本输出**：
- `2d_screening_results.json`：完整筛选结果
- `2d_material_candidates.json`：二维材料候选
- `2d_screening_summary.csv`：筛选结果摘要表
- `top_2d_candidates.csv`：按评分排序的顶级候选

### 分析报告文件
- `screening_summary_report.txt`：详细统计报告
- `figures/`：包含各种统计图表
  - `dimension_distribution.png`：维度分布图
  - `score_distribution.png`：评分分布图
  - `binding_energy_analysis.png`：结合能分析图
  - `element_analysis.png`：元素分析图

## 筛选参数配置

可以通过修改 `configs/2d_screening_config.yaml` 调整筛选参数：

```yaml
screening:
  max_elements: 3  # 最大元素种类
  max_binding_energy: 0.15  # 最大层间结合能 (eV/Å²)
  min_interlayer_distance: 2.5  # 最小层间距离 (Å)
  min_2d_score: 0.6  # 最小二维材料评分
```

## 评分系统

二维材料综合评分基于以下权重：
- TSA分析：40%
- 层状结构：30%
- 层间距离：20%
- 结合能：10%

## 论文验证

系统包含了论文中提到的已知二维材料用于验证：
- MoS₂, WSe₂, h-BN
- Bi₂Se₃, Bi₂Te₃
- InSe, GaS, GaSe
- SnS, SnSe, TiS₃
- TcS₂, TcSe₂

## 技术特点

1. **高精度识别**：结合多种算法提高识别准确性
2. **快速筛选**：快速版本处理速度达到 ~100 文件/秒
3. **双重验证**：既有快速版本也有详细分析版本
4. **可配置性**：所有关键参数可通过配置文件调整
5. **完整分析**：提供详细的统计分析和可视化
6. **论文对标**：基于最新研究成果的科学方法
7. **实验验证**：成功识别已知2D材料如GaTe等

## 注意事项

1. **数据质量**：确保输入的CIF文件结构合理
2. **计算资源**：大规模筛选可能需要较长时间
3. **结果验证**：建议对筛选结果进行进一步的DFT验证
4. **参数调优**：可根据具体需求调整筛选阈值

## 故障排除

### 常见问题

1. **内存不足**：
   - 减少批处理大小
   - 增加系统内存
   - 分批处理大型数据集

2. **依赖库错误**：
   - 检查pymatgen版本兼容性
   - 重新安装相关包

3. **CIF文件解析错误**：
   - 检查文件格式
   - 更新pymatgen到最新版本

4. **无筛选结果**：
   - 检查输入数据是否正确
   - 适当放宽筛选参数

## 引用

如使用该系统，请引用相关论文：
- 陈乐添, 陈安, 张旭, 周震. 二维材料的高通量筛选与光催化性能预测. 科学通报, 2021, 66: 606–624

## 联系方式

如有技术问题，请检查：
1. 系统日志输出
2. 配置文件设置
3. 输入数据格式
4. 环境依赖