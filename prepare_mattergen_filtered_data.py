#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为筛选后的材料准备MatterGen训练数据

读取筛选后的CIF文件和属性目标，准备MatterGen格式的数据

作者：Claude AI  
日期：2025-07-30
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_mattergen_data(
    filtered_dir="data/filtered",
    properties_file="data/predicted_properties.csv", 
    output_dir="data/mattergen_filtered"
):
    """
    准备MatterGen格式的筛选数据
    
    Args:
        filtered_dir: 筛选后CIF文件目录
        properties_file: 属性文件路径
        output_dir: 输出目录
    """
    
    filtered_dir = Path(filtered_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 读取属性数据
    if not Path(properties_file).exists():
        logger.error(f"属性文件不存在: {properties_file}")
        logger.info("请先运行: python create_property_targets.py")
        return
    
    logger.info(f"读取属性数据: {properties_file}")
    props_df = pd.read_csv(properties_file)
    
    # 获取筛选后的CIF文件
    cif_files = list(filtered_dir.glob("*.cif"))
    logger.info(f"找到 {len(cif_files)} 个CIF文件")
    
    # 创建MatterGen格式的数据
    mattergen_data = []
    
    for cif_file in cif_files:
        material_id = cif_file.stem
        
        # 查找对应的属性数据
        prop_row = props_df[props_df['material_id'] == material_id]
        if len(prop_row) == 0:
            logger.warning(f"未找到材料属性: {material_id}")
            continue
        
        prop_row = prop_row.iloc[0]
        
        # 读取CIF内容
        try:
            with open(cif_file, 'r', encoding='utf-8') as f:
                cif_content = f.read()
        except Exception as e:
            logger.error(f"读取CIF文件失败 {cif_file}: {e}")
            continue
        
        # 创建MatterGen记录
        record = {
            'material_id': material_id,
            'cif': cif_content,
            'formula': prop_row['formula'],
            'predicted_formation_energy': prop_row['predicted_formation_energy'],
            'predicted_energy_above_hull': prop_row['predicted_energy_above_hull'],
            'n_elements': int(prop_row['n_elements']),
            'is_2d': bool(prop_row['is_2d']),
            'target_stability_score': prop_row['target_stability_score']
        }
        
        mattergen_data.append(record)
    
    # 保存为CSV格式
    df = pd.DataFrame(mattergen_data)
    csv_path = output_dir / "materials.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"MatterGen数据已保存: {csv_path}")
    
    # 保存结构数据为JSON格式（MatterGen可能需要）
    structures_data = []
    properties_data = []
    
    for record in mattergen_data:
        structures_data.append({
            'material_id': record['material_id'],
            'cif': record['cif'],
            'formula': record['formula']
        })
        
        properties_data.append({
            'material_id': record['material_id'],
            'formation_energy_per_atom': record['predicted_formation_energy'],
            'energy_above_hull': record['predicted_energy_above_hull'],
            'target_stability_score': record['target_stability_score']
        })
    
    # 保存JSON文件
    with open(output_dir / "structures.json", 'w', encoding='utf-8') as f:
        json.dump(structures_data, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "properties.json", 'w', encoding='utf-8') as f:
        json.dump(properties_data, f, ensure_ascii=False, indent=2)
    
    # 创建数据集配置
    dataset_config = {
        "dataset_name": "filtered_2d_materials",
        "total_samples": len(mattergen_data),
        "property_columns": ["predicted_formation_energy", "predicted_energy_above_hull", "target_stability_score"],
        "target_property": "predicted_formation_energy",
        "stability_constraints": {
            "max_formation_energy": 0.0,
            "max_energy_above_hull": 0.1
        },
        "synthesizability_constraints": {
            "max_elements": 3
        },
        "description": "筛选后的合成性材料数据，用于条件生成训练"
    }
    
    with open(output_dir / "dataset_config.json", 'w', encoding='utf-8') as f:
        json.dump(dataset_config, f, ensure_ascii=False, indent=2)
    
    # 统计信息
    print("\n" + "="*50)
    print("MatterGen数据准备完成")
    print("="*50)
    print(f"总材料数: {len(mattergen_data)}")
    print(f"2D材料数: {sum(record['is_2d'] for record in mattergen_data)}")
    print(f"平均形成能: {np.mean([r['predicted_formation_energy'] for r in mattergen_data]):.3f} eV/atom")
    print(f"目标稳定材料数 (形成能<0): {sum(r['predicted_formation_energy'] < 0 for r in mattergen_data)}")
    print(f"输出目录: {output_dir}")
    print("="*50)
    
    logger.info("数据准备完成！可以开始MatterGen训练")
    
    return output_dir

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="准备MatterGen筛选数据")
    parser.add_argument("--filtered_dir", type=str, default="data/filtered",
                       help="筛选后材料目录")
    parser.add_argument("--properties_file", type=str, default="data/predicted_properties.csv",
                       help="属性文件路径")
    parser.add_argument("--output_dir", type=str, default="data/mattergen_filtered",
                       help="输出目录")
    
    args = parser.parse_args()
    
    prepare_mattergen_data(args.filtered_dir, args.properties_file, args.output_dir)

if __name__ == "__main__":
    main()