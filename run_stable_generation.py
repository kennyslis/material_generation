"""
简化的稳定材料生成执行脚本
使用配置文件驱动的材料生成和筛选流程
"""

import yaml
import argparse
from pathlib import Path
import sys
import os

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from generate_stable_materials import MaterialGenerator


class ConfigurableGenerator:
    """基于配置文件的材料生成器"""
    
    def __init__(self, config_path: str = "stable_generation_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"已加载配置文件: {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"错误: 找不到配置文件 {self.config_path}")
            print("使用默认配置...")
            return self.get_default_config()
        except Exception as e:
            print(f"配置文件加载错误: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            'generation': {
                'num_samples': 100,
                'batch_size': 8,
                'checkpoint_path': 'checkpoints/last.ckpt',
                'output_dir': 'stable_materials_output'
            },
            'stability_criteria': {
                'hull_energy_threshold': 0.1,
                'formation_energy_threshold': 0.0,
                'max_elements': 3,
                'min_2d_score': 0.7
            }
        }
    
    def create_generator(self) -> MaterialGenerator:
        """创建材料生成器"""
        gen_config = self.config['generation']
        generator = MaterialGenerator(
            checkpoint_path=gen_config['checkpoint_path'],
            output_dir=gen_config['output_dir']
        )
        
        # 更新筛选标准
        if 'stability_criteria' in self.config:
            generator.stability_criteria.update(self.config['stability_criteria'])
        
        # 更新HER元素配置
        if 'synthesis' in self.config:
            synthesis = self.config['synthesis']
            if 'her_active_elements' in synthesis:
                generator.her_elements['active'] = synthesis['her_active_elements']
            if 'chalcogen_elements' in synthesis:
                generator.her_elements['chalcogen'] = synthesis['chalcogen_elements']
            if 'support_elements' in synthesis:
                generator.her_elements['support'] = synthesis['support_elements']
        
        return generator
    
    def run(self):
        """执行材料生成流程"""
        print("=== 基于配置的稳定材料生成 ===")
        
        # 显示配置信息
        gen_config = self.config['generation']
        stability_config = self.config.get('stability_criteria', {})
        
        print(f"生成数量: {gen_config.get('num_samples', 100)}")
        print(f"批处理大小: {gen_config.get('batch_size', 8)}")
        print(f"输出目录: {gen_config.get('output_dir', 'stable_materials_output')}")
        print(f"Hull能量阈值: {stability_config.get('hull_energy_threshold', 0.1)} eV/atom")
        print(f"形成能阈值: {stability_config.get('formation_energy_threshold', 0.0)} eV/atom")
        print(f"最大元素数: {stability_config.get('max_elements', 3)}")
        print(f"最小2D评分: {stability_config.get('min_2d_score', 0.7)}")
        
        # 创建生成器
        generator = self.create_generator()
        
        # 执行生成流程
        try:
            # 1. 生成材料
            print("\n--- 步骤1: 生成材料 ---")
            cif_files = generator.generate_materials_mattergen(
                num_samples=gen_config.get('num_samples', 100),
                batch_size=gen_config.get('batch_size', 8)
            )
            
            if not cif_files:
                print("错误: 没有生成任何材料文件")
                return False
            
            # 2. 分析材料
            print("\n--- 步骤2: 分析材料 ---")
            original_df = generator.analyze_materials(cif_files)
            
            # 3. 筛选材料
            print("\n--- 步骤3: 筛选材料 ---")
            filtered_df = generator.filter_materials(original_df)
            
            # 4. 生成报告
            print("\n--- 步骤4: 生成报告 ---")
            generator.generate_report(original_df, filtered_df)
            
            print("\n=== 生成完成! ===")
            self.print_summary(original_df, filtered_df, gen_config['output_dir'])
            
            return True
            
        except Exception as e:
            print(f"生成过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_summary(self, original_df, filtered_df, output_dir):
        """打印结果摘要"""
        print(f"\n生成结果摘要:")
        print(f"   总生成材料: {len(original_df)}")
        print(f"   通过筛选: {len(filtered_df)}")
        print(f"   成功率: {len(filtered_df)/len(original_df)*100:.1f}%")
        
        print(f"\n输出文件:")
        print(f"   输出目录: {output_dir}")
        print(f"   CIF文件: {output_dir}/cif_files/")
        print(f"   分析结果: {output_dir}/analysis/generated_materials_analysis.csv")
        print(f"   筛选结果: {output_dir}/filtered/stable_synthesizable_materials.csv")
        print(f"   详细报告: {output_dir}/analysis/generation_report.json")
        
        if len(filtered_df) > 0:
            print(f"\n推荐材料:")
            for i, material in enumerate(filtered_df.head(3).itertuples(), 1):
                score = getattr(material, 'overall_score', 0)
                formula = getattr(material, 'formula', 'Unknown')
                print(f"   {i}. {formula} (综合评分: {score:.3f})")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于配置的稳定材料生成")
    parser.add_argument("--config", type=str, default="stable_generation_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--num-samples", type=int, help="覆盖配置中的生成数量")
    parser.add_argument("--output-dir", type=str, help="覆盖配置中的输出目录")
    
    args = parser.parse_args()
    
    # 创建可配置生成器
    generator = ConfigurableGenerator(args.config)
    
    # 命令行参数覆盖配置
    if args.num_samples:
        generator.config['generation']['num_samples'] = args.num_samples
    if args.output_dir:
        generator.config['generation']['output_dir'] = args.output_dir
    
    # 运行生成流程
    success = generator.run()
    
    if success:
        print("\n材料生成完成!")
        exit(0)
    else:
        print("\n材料生成失败!")
        exit(1)


if __name__ == "__main__":
    main()