"""
准备MatterGen微调训练数据
将筛选后的优质材料转换为训练格式
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
import yaml

class TrainingDataPreparer:
    """训练数据准备器"""
    
    def __init__(self, 
                 filtered_dir="mattergen_output/filtered",
                 training_data_dir="training_data"):
        self.filtered_dir = Path(filtered_dir)
        self.training_data_dir = Path(training_data_dir)
        self.training_data_dir.mkdir(exist_ok=True)
        
        # 创建训练数据子目录
        (self.training_data_dir / "structures").mkdir(exist_ok=True)
        (self.training_data_dir / "properties").mkdir(exist_ok=True)
    
    def collect_high_quality_materials(self, dft_results_file: str) -> pd.DataFrame:
        """收集高质量材料数据
        
        Args:
            dft_results_file: DFT计算结果CSV文件，包含formation_energy等列
        """
        print("收集高质量材料数据...")
        
        # 读取DFT计算结果
        dft_df = pd.read_csv(dft_results_file)
        
        # 应用最终筛选标准
        high_quality = dft_df[
            (dft_df['num_elements'] <= 3) &
            (dft_df['energy_above_hull'] < 0.1) &
            (dft_df['formation_energy_per_atom'] < 0.0)
        ].copy()
        
        print(f"筛选出 {len(high_quality)} 个高质量材料")
        return high_quality
    
    def prepare_mattergen_format(self, high_quality_df: pd.DataFrame):
        """准备MatterGen训练格式数据"""
        print("准备MatterGen训练格式...")
        
        structures_data = []
        properties_data = []
        
        for idx, row in high_quality_df.iterrows():
            try:
                # 读取结构
                cif_file = row['cif_file']
                structure = Structure.from_file(cif_file)
                
                # 结构数据
                structure_data = {
                    "material_id": row['material_id'],
                    "lattice": structure.lattice.matrix.tolist(),
                    "species": [str(site.specie) for site in structure],
                    "coords": structure.cart_coords.tolist(),
                    "formula": structure.composition.reduced_formula,
                    "num_atoms": len(structure),
                    "volume": structure.volume,
                    "density": structure.density
                }
                structures_data.append(structure_data)
                
                # 属性数据
                properties_data.append({
                    "material_id": row['material_id'],
                    "energy_above_hull": row['energy_above_hull'],
                    "formation_energy_per_atom": row['formation_energy_per_atom'],
                    "chemical_system": "-".join(sorted(set([str(site.specie) for site in structure]))),
                    "num_elements": row['num_elements'],
                    "space_group": row.get('space_group', 1),
                    "is_2d_candidate": True  # 标记为二维候选
                })
                
            except Exception as e:
                print(f"处理 {row['material_id']} 时出错: {e}")
                continue
        
        # 保存数据
        with open(self.training_data_dir / "structures" / "high_quality_structures.json", "w") as f:
            json.dump(structures_data, f, indent=2)
            
        with open(self.training_data_dir / "properties" / "high_quality_properties.json", "w") as f:
            json.dump(properties_data, f, indent=2)
        
        print(f"保存了 {len(structures_data)} 个高质量训练样本")
        return len(structures_data)
    
    def create_training_csv(self, high_quality_df: pd.DataFrame):
        """创建MatterGen训练CSV格式"""
        print("创建训练CSV文件...")
        
        # 添加必要的训练标签
        training_df = high_quality_df.copy()
        training_df['split'] = 'train'  # 标记为训练集
        training_df['quality_score'] = (
            (1.0 - training_df['energy_above_hull'] / 0.1) * 0.4 +  # hull能量权重
            (1.0 + training_df['formation_energy_per_atom'] / -1.0) * 0.4 +  # 形成能权重  
            (1.0 / training_df['num_elements']) * 0.2  # 元素简单性权重
        )
        
        # 保存训练CSV
        csv_path = self.training_data_dir / "high_quality_materials.csv"
        training_df.to_csv(csv_path, index=False)
        
        print(f"训练CSV保存至: {csv_path}")
        print(f"平均质量评分: {training_df['quality_score'].mean():.3f}")
        
        return str(csv_path)
    
    def create_finetune_config(self, num_samples: int):
        """创建微调配置文件"""
        print("创建微调配置文件...")
        
        config = {
            "base_model": "chemical_system_energy_above_hull",
            "finetune_config": {
                "pretrained_ckpt_path": "checkpoints/chemical_system_energy_above_hull",
                "data": {
                    "root_dir": str(self.training_data_dir),
                    "properties": [
                        "energy_above_hull",
                        "formation_energy_per_atom", 
                        "chemical_system"
                    ],
                    "batch_size": 8,
                    "num_workers": 0
                },
                "model": {
                    "condition_on_property": True,
                    "property_guidance_scale": 1.0
                },
                "training": {
                    "max_epochs": 50,
                    "learning_rate": 1e-5,
                    "warmup_steps": 100,
                    "save_top_k": 3,
                    "patience": 10
                },
                "output": {
                    "checkpoint_dir": "checkpoints/finetuned_model",
                    "log_dir": "logs/finetune"
                }
            },
            "data_info": {
                "num_high_quality_samples": num_samples,
                "constraints_applied": [
                    "num_elements <= 3",
                    "energy_above_hull < 0.1 eV/atom",
                    "formation_energy_per_atom < 0.0 eV/atom"
                ]
            }
        }
        
        config_path = self.training_data_dir / "finetune_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"微调配置保存至: {config_path}")
        return str(config_path)
    
    def prepare_all(self, dft_results_file: str):
        """执行完整的数据准备流程"""
        print("=== 开始准备微调训练数据 ===")
        
        # 1. 收集高质量材料
        high_quality_df = self.collect_high_quality_materials(dft_results_file)
        
        if len(high_quality_df) == 0:
            print("警告: 没有找到符合标准的高质量材料!")
            return
        
        # 2. 准备MatterGen格式
        num_samples = self.prepare_mattergen_format(high_quality_df)
        
        # 3. 创建训练CSV
        csv_path = self.create_training_csv(high_quality_df)
        
        # 4. 创建微调配置
        config_path = self.create_finetune_config(num_samples)
        
        print("\n=== 数据准备完成 ===")
        print(f"高质量样本数: {num_samples}")
        print(f"训练数据目录: {self.training_data_dir}")
        print(f"下一步: 使用 {config_path} 进行模型微调")
        
        return {
            "num_samples": num_samples,
            "csv_path": csv_path,
            "config_path": config_path,
            "training_dir": str(self.training_data_dir)
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="准备MatterGen微调训练数据")
    parser.add_argument("--dft-results", type=str, required=True,
                       help="DFT计算结果CSV文件路径")
    parser.add_argument("--output-dir", type=str, default="training_data",
                       help="训练数据输出目录")
    
    args = parser.parse_args()
    
    preparer = TrainingDataPreparer(training_data_dir=args.output_dir)
    results = preparer.prepare_all(args.dft_results)
    
    if results:
        print(f"\n微调训练命令:")
        print(f"mattergen-train --config-path {results['training_dir']} --config-name finetune_config")


if __name__ == "__main__":
    main()