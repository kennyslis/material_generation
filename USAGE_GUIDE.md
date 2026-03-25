# 二维材料筛选系统使用指南

## 快速开始

### 1. 环境检查
确保已安装必要的Python包：
```bash
pip install pymatgen pandas matplotlib seaborn numpy
```

### 2. 数据准备
确保 `data/raw/` 目录包含Materials Project的CIF文件。

### 3. 运行筛选流程

#### 步骤1：合成性预筛选
```bash
python filter_synthesizable_materials_simple.py
```
- **输入**：`data/raw/` 中的原始CIF文件
- **输出**：`data/filtered/` 中筛选后的≤3元素材料
- **预计耗时**：根据文件数量，可能需要几小时

#### 步骤2：二维性筛选（推荐快速版本）
```bash
python run_fast_2d_screening.py
```
- **输入**：`data/filtered/` 中的预筛选材料
- **输出**：`data/fast_2d_materials/` 中的二维候选材料
- **处理速度**：~100 文件/秒
- **预计耗时**：几分钟到几十分钟

#### 步骤3：结果分析（可选）
```bash
python analyze_2d_results.py
```
- **功能**：生成统计报告和可视化图表

## 版本选择

### 快速版本 vs 完整版本

| 特性 | 快速版本 | 完整版本 |
|------|----------|----------|
| 处理速度 | ~100 文件/秒 | ~1 文件/秒 |
| 算法复杂度 | 简化几何分析 | 拓扑缩放算法+图分析 |
| 准确性 | 良好 | 更高 |
| 推荐场景 | 大规模筛选 | 详细研究 |

### 何时使用快速版本？
- ✅ 处理大量材料数据（>10000个文件）
- ✅ 初步筛选阶段
- ✅ 需要快速获得结果
- ✅ 计算资源有限

### 何时使用完整版本？
- ✅ 精确分析少量材料
- ✅ 需要详细的结构分析
- ✅ 科研论文级别的精度要求
- ✅ 有充足的计算时间

## 输出文件解读

### 快速版本输出

**文件位置**：`data/fast_2d_materials/`

1. **`fast_2d_candidates.json`** - 二维候选材料
   ```json
   {
     "filename": "mp_mp-xxx.cif",
     "formula": "GaTe",
     "2d_analysis": {
       "is_2d": true,
       "confidence": 0.328,
       "max_aspect_ratio": 4.3
     },
     "score": 0.597
   }
   ```

2. **`fast_2d_summary.csv`** - 表格格式摘要
   - 可用Excel或其他工具打开
   - 包含所有关键指标

### 关键指标解释

- **`max_aspect_ratio`**：最大轴长比，>2.0表示可能为层状
- **`confidence`**：算法置信度，0-1之间，越高越可信
- **`stability_score`**：稳定性评分，0-1之间
- **`score`**：综合评分，0-1之间，>0.4为候选

## 参数调优

### 修改筛选标准

编辑 `src/data/filter_2d_materials_fast.py` 中的参数：

```python
# 筛选参数
self.max_elements = 3           # 最大元素种类
self.min_aspect_ratio = 1.5     # 最小长宽比
self.max_layer_thickness = 0.4  # 最大层厚度

# 判定标准
is_2d_candidate = (
    twoD_analysis['is_2d'] and
    twoD_analysis['confidence'] > 0.3 and  # 置信度阈值
    stability['stability_score'] > 0.2 and  # 稳定性阈值
    score > 0.4  # 评分阈值
)
```

### 常见调优策略

**更严格筛选**（减少假阳性）：
- 提高 `min_aspect_ratio` 到 2.5
- 提高置信度阈值到 0.5
- 提高评分阈值到 0.6

**更宽松筛选**（增加候选数量）：
- 降低 `min_aspect_ratio` 到 1.3
- 降低置信度阈值到 0.2
- 降低评分阈值到 0.3

## 验证结果

### 已知二维材料验证

系统应该能识别以下已知2D材料：
- **石墨烯类**：C
- **过渡金属二硫族**：MoS₂, WS₂, WSe₂
- **三-六族化合物**：GaS, GaSe, InSe, GaTe
- **黑磷类**：各种磷的同素异形体
- **六方氮化硼**：h-BN

### 结果可信度判断

1. **高可信度候选** (score > 0.7)：
   - 很可能是真正的2D材料
   - 建议优先进行实验验证

2. **中等可信度候选** (0.4 < score < 0.7)：
   - 需要进一步分析
   - 可考虑DFT计算验证

3. **低可信度候选** (score < 0.4)：
   - 通常被筛除
   - 可能存在假阳性

## 常见问题解决

### 1. 处理速度慢
- 确保使用快速版本
- 检查系统内存和CPU使用率
- 考虑分批处理大型数据集

### 2. 内存不足
```bash
# 监控内存使用
python -u run_fast_2d_screening.py
```
- 重启Python进程清理内存
- 增加系统虚拟内存

### 3. 找不到候选材料
- 检查输入数据质量
- 适当放宽筛选参数
- 确认第一阶段筛选成功完成

### 4. CIF文件解析错误
- 检查文件格式和完整性
- 更新pymatgen到最新版本
- 删除损坏的CIF文件

## 性能优化建议

### 系统配置
- **CPU**：多核处理器，推荐4核以上
- **内存**：8GB以上，推荐16GB
- **存储**：SSD硬盘提高I/O性能

### 批处理策略
```python
# 分批处理大数据集
input_files = list(input_dir.glob("*.cif"))
batch_size = 1000

for i in range(0, len(input_files), batch_size):
    batch = input_files[i:i+batch_size]
    # 处理当前批次
```

## 下一步应用

筛选出的二维材料候选可用于：

1. **MatterGen训练**：作为高质量训练数据
2. **DFT验证**：进行精确的第一性原理计算
3. **实验合成**：指导实际材料制备
4. **性质预测**：进一步的功能性质计算

## 技术支持

如遇到技术问题，请：
1. 检查错误日志输出
2. 确认环境配置正确
3. 参考README文档
4. 查看测试脚本示例