"""
MatterGen迭代微调训练脚本
实现循环改进策略：生成 → 筛选 → 训练 → 改进
"""

import os
import json
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
import yaml
import shutil

class IterativeTrainer:
    """迭代训练器"""
    
    def __init__(self, base_dir="iterative_training"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 创建迭代目录结构
        self.iterations_dir = self.base_dir / "iterations"
        self.iterations_dir.mkdir(exist_ok=True)
        
        self.current_iteration = 0
        self.training_history = []
    
    def start_iteration(self, iteration_num: int):
        """开始新的迭代"""
        self.current_iteration = iteration_num
        self.iteration_dir = self.iterations_dir / f"iteration_{iteration_num:02d}"
        self.iteration_dir.mkdir(exist_ok=True)
        
        # 创建迭代子目录
        (self.iteration_dir / "generated").mkdir(exist_ok=True)
        (self.iteration_dir / "filtered").mkdir(exist_ok=True)
        (self.iteration_dir / "training_data").mkdir(exist_ok=True)
        (self.iteration_dir / "checkpoints").mkdir(exist_ok=True)
        (self.iteration_dir / "logs").mkdir(exist_ok=True)
        
        print(f"=== 开始第 {iteration_num} 轮迭代训练 ===")
        print(f"迭代目录: {self.iteration_dir}")
    
    def generate_materials(self, model_checkpoint: str, num_samples: int = 200):
        """使用当前模型生成材料"""
        print(f"\n--- 步骤1: 生成 {num_samples} 个候选材料 ---")
        
        output_dir = self.iteration_dir / "generated"
        
        cmd = [
            "python", "run_mattergen_generation.py",
            "--checkpoint", model_checkpoint,
            "--energy-above-hull", "0.1",
            "--max-elements", "3", 
            "--num-samples", str(num_samples),
            "--batch-size", "16",
            "--output-dir", str(output_dir)
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 材料生成完成")
            return output_dir / "cif_files"
        else:
            print(f"❌ 生成失败: {result.stderr}")
            raise RuntimeError("材料生成失败")
    
    def dft_calculation_placeholder(self, cif_dir: Path):
        """DFT计算占位符 - 你需要替换为实际的DFT计算"""
        print("\n--- 步骤2: DFT计算Formation energy ---")
        print("⚠️  注意: 这里需要你实现实际的DFT计算!")
        print("     建议使用VASP、Quantum ESPRESSO或ASE等工具")
        
        # 创建模拟的DFT结果 - 实际使用时删除这部分
        cif_files = list(cif_dir.glob("*.cif"))
        dft_results = []
        
        for cif_file in cif_files:
            # 这里是模拟数据 - 实际要用DFT计算
            import random
            formation_energy = random.uniform(-2.0, 1.0)  # eV/atom
            
            dft_results.append({
                "material_id": cif_file.stem,
                "cif_file": str(cif_file),
                "formation_energy_per_atom": formation_energy,
                "energy_above_hull": random.uniform(0.0, 0.15),
                "num_elements": random.randint(2, 3),
                "dft_calculated": True
            })
        
        # 保存DFT结果
        dft_results_file = self.iteration_dir / "dft_results.csv"
        pd.DataFrame(dft_results).to_csv(dft_results_file, index=False)
        
        print(f"✅ DFT结果保存至: {dft_results_file}")
        return str(dft_results_file)
    
    def filter_high_quality(self, dft_results_file: str):
        """筛选高质量材料"""
        print("\n--- 步骤3: 筛选高质量材料 ---")
        
        df = pd.read_csv(dft_results_file)
        
        # 应用严格筛选标准
        high_quality = df[
            (df['num_elements'] <= 3) &
            (df['energy_above_hull'] < 0.1) &
            (df['formation_energy_per_atom'] < 0.0)
        ].copy()
        
        print(f"原始材料: {len(df)} 个")
        print(f"高质量材料: {len(high_quality)} 个")
        print(f"筛选成功率: {len(high_quality)/len(df)*100:.1f}%")
        
        # 保存筛选结果
        filtered_file = self.iteration_dir / "filtered" / "high_quality_materials.csv"
        high_quality.to_csv(filtered_file, index=False)
        
        return str(filtered_file), len(high_quality)
    
    def prepare_training_data(self, filtered_file: str):
        """准备训练数据"""
        print("\n--- 步骤4: 准备训练数据 ---")
        
        training_dir = self.iteration_dir / "training_data"
        
        cmd = [
            "python", "prepare_training_data.py",
            "--dft-results", filtered_file,
            "--output-dir", str(training_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 训练数据准备完成")
            return training_dir / "finetune_config.yaml"
        else:
            print(f"❌ 数据准备失败: {result.stderr}")
            raise RuntimeError("训练数据准备失败")
    
    def finetune_model(self, config_file: Path, base_checkpoint: str):
        """微调模型"""
        print("\n--- 步骤5: 微调模型 ---")
        
        checkpoint_dir = self.iteration_dir / "checkpoints"
        log_dir = self.iteration_dir / "logs"
        
        # 创建微调配置
        finetune_config = {
            "defaults": ["_self_"],
            "finetune": {
                "pretrained_ckpt_path": base_checkpoint,
                "target_properties": ["energy_above_hull", "formation_energy_per_atom"],
                "learning_rate": 1e-5,
                "max_epochs": 30,
                "batch_size": 8,
                "patience": 5
            },
            "trainer": {
                "max_epochs": 30,
                "precision": "16-mixed",
                "devices": 1,
                "log_every_n_steps": 10,
                "check_val_every_n_epoch": 2
            },
            "data": {
                "csv_path": str(config_file.parent / "high_quality_materials.csv"),
                "batch_size": 8,
                "num_workers": 0
            },
            "callbacks": {
                "checkpoint": {
                    "dirpath": str(checkpoint_dir),
                    "filename": f"iter_{self.current_iteration:02d}-{{epoch}}-{{val_loss:.3f}}",
                    "save_top_k": 2,
                    "monitor": "val_loss"
                }
            }
        }
        
        config_path = self.iteration_dir / "finetune_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(finetune_config, f, default_flow_style=False)
        
        # 执行微调
        cmd = [
            "mattergen-train",
            "--config-path", str(self.iteration_dir),
            "--config-name", "finetune_config"
        ]
        
        print(f"执行微调命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
        
        if result.returncode == 0:
            print("✅ 模型微调完成")
            # 找到最佳checkpoint
            best_ckpt = self.find_best_checkpoint(checkpoint_dir)
            return best_ckpt
        else:
            print(f"❌ 微调失败: {result.stderr}")
            return None
    
    def find_best_checkpoint(self, checkpoint_dir: Path):
        """找到最佳checkpoint"""
        ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
        if ckpt_files:
            # 简单选择最新的checkpoint，实际可以基于validation loss选择
            best_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
            print(f"最佳checkpoint: {best_ckpt}")
            return str(best_ckpt)
        return None
    
    def evaluate_iteration(self, num_high_quality: int, base_quality: int = None):
        """评估迭代效果"""
        print(f"\n--- 步骤6: 迭代效果评估 ---")
        
        if base_quality is None:
            improvement = "N/A (首次迭代)"
        else:
            improvement = f"{(num_high_quality - base_quality) / base_quality * 100:+.1f}%"
        
        iteration_stats = {
            "iteration": self.current_iteration,
            "timestamp": datetime.now().isoformat(),
            "high_quality_materials": num_high_quality,
            "improvement": improvement,
            "directory": str(self.iteration_dir)
        }
        
        self.training_history.append(iteration_stats)
        
        print(f"迭代 {self.current_iteration} 结果:")
        print(f"  高质量材料数: {num_high_quality}")
        print(f"  相比上轮改进: {improvement}")
        
        # 保存历史记录
        history_file = self.base_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return iteration_stats
    
    def run_iteration(self, model_checkpoint: str, num_samples: int = 200, base_quality: int = None):
        """运行完整的迭代"""
        iteration_num = len(self.training_history) + 1
        self.start_iteration(iteration_num)
        
        try:
            # 1. 生成材料
            cif_dir = self.generate_materials(model_checkpoint, num_samples)
            
            # 2. DFT计算 (占位符)
            dft_results_file = self.dft_calculation_placeholder(cif_dir)
            
            # 3. 筛选高质量材料
            filtered_file, num_high_quality = self.filter_high_quality(dft_results_file)
            
            if num_high_quality == 0:
                print("❌ 没有找到高质量材料，跳过训练")
                return None
            
            # 4. 准备训练数据
            config_file = self.prepare_training_data(filtered_file)
            
            # 5. 微调模型
            new_checkpoint = self.finetune_model(config_file, model_checkpoint)
            
            # 6. 评估效果
            stats = self.evaluate_iteration(num_high_quality, base_quality)
            
            print(f"\n✅ 第 {iteration_num} 轮迭代完成!")
            if new_checkpoint:
                print(f"新模型checkpoint: {new_checkpoint}")
                return new_checkpoint, num_high_quality
            else:
                return model_checkpoint, num_high_quality  # 使用原模型
                
        except Exception as e:
            print(f"❌ 迭代失败: {e}")
            return None
    
    def run_multiple_iterations(self, initial_checkpoint: str, max_iterations: int = 5, target_quality: int = 50):
        """运行多轮迭代训练"""
        print(f"=== 开始多轮迭代训练 ===")
        print(f"目标: {max_iterations} 轮迭代或达到 {target_quality} 个高质量材料")
        
        current_checkpoint = initial_checkpoint
        last_quality = None
        
        for i in range(max_iterations):
            print(f"\n{'='*50}")
            print(f"第 {i+1}/{max_iterations} 轮迭代")
            print(f"{'='*50}")
            
            result = self.run_iteration(current_checkpoint, base_quality=last_quality)
            
            if result is None:
                print("迭代失败，停止训练")
                break
            
            new_checkpoint, quality_count = result
            
            # 更新checkpoint
            if new_checkpoint != current_checkpoint:
                current_checkpoint = new_checkpoint
                print(f"✅ 模型已更新: {current_checkpoint}")
            
            last_quality = quality_count
            
            # 检查是否达到目标
            if quality_count >= target_quality:
                print(f"🎉 达到目标质量 ({quality_count} >= {target_quality})!")
                break
        
        print(f"\n=== 迭代训练完成 ===")
        print(f"最终模型: {current_checkpoint}")
        print(f"训练历史: {self.base_dir / 'training_history.json'}")
        
        return current_checkpoint


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MatterGen迭代微调训练")
    parser.add_argument("--initial-checkpoint", type=str, 
                       default="chemical_system_energy_above_hull",
                       help="初始模型checkpoint")
    parser.add_argument("--iterations", type=int, default=3,
                       help="最大迭代次数")
    parser.add_argument("--target-quality", type=int, default=30,
                       help="目标高质量材料数量")
    parser.add_argument("--samples-per-iteration", type=int, default=100,
                       help="每轮生成的材料数量")
    
    args = parser.parse_args()
    
    trainer = IterativeTrainer()
    final_checkpoint = trainer.run_multiple_iterations(
        initial_checkpoint=args.initial_checkpoint,
        max_iterations=args.iterations,
        target_quality=args.target_quality
    )
    
    print(f"\n🎉 训练完成! 最终模型: {final_checkpoint}")


if __name__ == "__main__":
    main()