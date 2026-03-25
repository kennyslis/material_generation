# Real Public Dataset Setup

本项目已经支持接入真实公开二维材料数据。推荐优先使用 2DMatPedia 整库 JSON。

## 1. 下载数据

官方入口：`https://www.2dmatpedia.org/`。`n`n从官网的数据访问入口下载整库 JSON 或压缩版 JSON.GZ。

## 2. 放置文件

把下载文件放到以下任一位置：

- `dataset/raw/2dmatpedia.json`
- `dataset/raw/2dmatpedia.json.gz`
- `dataset/raw/2dmatpedia_entries.json`
- `dataset/raw/2dmatpedia_entries.json.gz`
- `dataset/raw/twodmatpedia.json`
- `dataset/raw/twodmatpedia.json.gz`

也可以用环境变量指定任意路径：

```powershell
$env:TWOD_DATA_JSON = "E:\机器学习项目\project\dataset\raw\2dmatpedia.json"
```

## 3. 清理旧缓存

如果之前已经跑过 surrogate 数据，建议先删掉旧缓存，避免混淆：

```powershell
cd E:\机器学习项目\project
Remove-Item -Force dataset\processed\real2d_* -ErrorAction SilentlyContinue
```

如果想把旧的 surrogate 缓存也一起清掉：

```powershell
Remove-Item -Force dataset\processed\surrogate_* -ErrorAction SilentlyContinue
```

## 4. 检查是否识别到真实数据

执行：

```powershell
cd E:\机器学习项目\project
conda run -n torch118 python -c "from dataset.material_dataset import MaterialDataset; ds=MaterialDataset('dataset', split='train', num_samples=8, dataset_source='real', real_data_path='dataset/raw/2dmatpedia.json'); print('len=', len(ds)); print('processed=', ds.processed_path); print('source=', ds.records[0].metadata.get('source')); print('material_id=', ds.records[0].metadata.get('material_id'))"
```

如果成功，通常会看到：

- `processed= dataset\processed\real2d_train_materials.json`
- `source= 2dmatpedia`
- `material_id= ...`

如果仍然显示 `surrogate`，说明真实数据文件没有被正确识别。

## 5. 训练真实数据版本

```powershell
cd E:\机器学习项目\project
conda run -n torch118 python train.py --output-dir . --dataset-source real --real-data-path dataset/raw/2dmatpedia.json --max-nodes 12 --epochs 60 --batch-size 32 --train-samples 320 --val-samples 96
```

如果你的真实数据规模更大，建议先用：

```powershell
conda run -n torch118 python train.py --output-dir . --dataset-source real --real-data-path dataset/raw/2dmatpedia.json --max-nodes 12 --epochs 60 --batch-size 32 --train-samples 2000 --val-samples 400
```

## 6. 测试真实数据版本

```powershell
cd E:\机器学习项目\project
conda run -n torch118 python test.py --output-dir . --dataset-source real --real-data-path dataset/raw/2dmatpedia.json --max-nodes 12 --test-samples 400 --num-candidates 64 --top-k 10
```

## 7. 训练后检查清单

训练完成后检查以下文件：

- `checkpoints/best_model.pt`
- `checkpoints/last_model.pt`
- `results/loss_curve.png`
- `results/training_history.json`

测试完成后检查：

- `results/her_performance.png`
- `results/stability_curve.png`
- `results/generated_structures.png`
- `results/comparison_metrics.json`
- `generated_structures/candidate_01.json` 到 `candidate_10.json`

## 8. 如何确认这次结果确实来自真实数据

检查这两个点：

1. 缓存文件名应为：
- `dataset/processed/real2d_train_materials.json`
- `dataset/processed/real2d_val_materials.json`
- `dataset/processed/real2d_test_materials.json`

2. 缓存文件里 `metadata.source` 应为：
- `2dmatpedia`

可用命令快速检查：

```powershell
Get-Content dataset\processed\real2d_train_materials.json -TotalCount 40
```

## 9. 口径建议

面试时建议这样说：

- 结构数据来自真实公开二维材料数据库 2DMatPedia。
- 稳定性相关字段优先使用公开数据库中的 exfoliation energy、decomposition energy、band gap 等信息。
- HER 的 `ΔG_H` 在当前工程版本中仍采用代理监督，以确保在有限时间内完成完整生成与优化链路。
- 代码已经预留 JARVIS-DFT 和 C2DB 的进一步融合空间。

