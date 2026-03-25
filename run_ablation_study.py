"""
消融实验脚本 - 验证TSA权重配置的合理性
用于回答审稿人关于权重设置依据的问题

实验设计:
1. 测试三种极端权重配置
2. 随机选择一定数量的材料进行筛选
3. 生成人工评估报告
4. 统计各配置的准确率
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict
import pandas as pd
from datetime import datetime

import sys
sys.path.append('src')
from data.filter_2d_materials_advanced import AdvancedTwoDMaterialScreener


class AblationStudyRunner:
    """消融实验运行器"""

    def __init__(self, input_dir: str, output_dir: str, sample_size: int = 50):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size

        # 定义三种极端权重配置（TSA内部）
        self.tsa_weight_configs = {
            'Config_A_Topology_Only': {
                'name': '配置A: 仅拓扑特征',
                'weights': {'topology': 1.0, 'graph': 0.0, 'geometric': 0.0}
            },
            'Config_B_Graph_Only': {
                'name': '配置B: 仅图论特征',
                'weights': {'topology': 0.0, 'graph': 1.0, 'geometric': 0.0}
            },
            'Config_C_Geometric_Only': {
                'name': '配置C: 仅几何特征',
                'weights': {'topology': 0.0, 'graph': 0.0, 'geometric': 1.0}
            },
            'Config_Default': {
                'name': '配置D: 默认权重 (40/30/30)',
                'weights': {'topology': 0.40, 'graph': 0.30, 'geometric': 0.30}
            }
        }

    def select_random_samples(self) -> List[Path]:
        """随机选择样本文件"""
        all_cif_files = list(self.input_dir.glob("*.cif"))

        if len(all_cif_files) <= self.sample_size:
            print(f"警告: 文件总数({len(all_cif_files)}) <= 样本数({self.sample_size}), 使用全部文件")
            return all_cif_files

        samples = random.sample(all_cif_files, self.sample_size)
        print(f"从{len(all_cif_files)}个文件中随机选择{len(samples)}个样本")
        return samples

    def run_single_config(self, config_name: str, config_info: Dict, sample_files: List[Path]) -> Dict:
        """运行单个权重配置的筛选"""
        print(f"\n{'='*60}")
        print(f"运行配置: {config_info['name']}")
        print(f"权重: {config_info['weights']}")
        print(f"{'='*60}")

        # 创建临时目录
        temp_output_dir = self.output_dir / config_name
        temp_output_dir.mkdir(parents=True, exist_ok=True)

        # 创建筛选器并设置权重
        screener = AdvancedTwoDMaterialScreener(
            str(self.input_dir),
            str(temp_output_dir),
            use_dft=False,
            use_topology=True
        )

        # 设置TSA权重配置
        screener.tsa_weight_config = config_info['weights']

        # 处理每个样本
        results = []
        for cif_file in sample_files:
            result = screener.process_single_material(cif_file)
            results.append(result)

        # 统计结果
        candidates = [r for r in results if r.get('is_2d_candidate', False)]

        summary = {
            'config_name': config_name,
            'config_label': config_info['name'],
            'weights': config_info['weights'],
            'total_processed': len(results),
            'num_candidates': len(candidates),
            'candidate_rate': len(candidates) / len(results) if results else 0,
            'results': results,
            'candidate_files': [r['filename'] for r in candidates]
        }

        print(f"\n结果统计:")
        print(f"- 处理文件数: {summary['total_processed']}")
        print(f"- 识别为2D: {summary['num_candidates']}")
        print(f"- 2D比例: {summary['candidate_rate']:.1%}")

        return summary

    def run_all_configs(self) -> Dict:
        """运行所有权重配置"""
        print(f"开始消融实验")
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"样本数量: {self.sample_size}")

        # 选择样本
        sample_files = self.select_random_samples()

        # 保存样本列表
        sample_list_file = self.output_dir / "sample_list.json"
        with open(sample_list_file, 'w') as f:
            json.dump([str(f) for f in sample_files], f, indent=2)

        # 运行所有配置
        all_results = {}
        for config_name, config_info in self.tsa_weight_configs.items():
            summary = self.run_single_config(config_name, config_info, sample_files)
            all_results[config_name] = summary

        return all_results

    def generate_evaluation_report(self, all_results: Dict) -> None:
        """生成人工评估报告"""
        print(f"\n{'='*60}")
        print("生成人工评估报告")
        print(f"{'='*60}")

        # 创建评估模板
        evaluation_template = []

        for config_name, summary in all_results.items():
            candidates = summary['candidate_files']

            for filename in candidates:
                evaluation_template.append({
                    'config_name': config_name,
                    'config_label': summary['config_label'],
                    'filename': filename,
                    'is_true_2d': '',  # 待人工填写 (填写 yes/no)
                    'confidence': '',  # 待人工填写 (填写 high/medium/low)
                    'notes': ''  # 待人工填写 (备注)
                })

        # 保存为Excel便于人工评估
        df = pd.DataFrame(evaluation_template)
        excel_file = self.output_dir / "human_evaluation_template.xlsx"
        df.to_excel(excel_file, index=False)

        print(f"✓ 人工评估模板已生成: {excel_file}")
        print(f"  共{len(evaluation_template)}个候选材料待评估")
        print(f"\n请在Excel中填写以下列:")
        print(f"  - is_true_2d: 填写 yes/no (该材料是否真的是二维材料)")
        print(f"  - confidence: 填写 high/medium/low (你对判断的置信度)")
        print(f"  - notes: 填写备注信息")

        # 生成对比分析表
        comparison_data = []
        for config_name, summary in all_results.items():
            comparison_data.append({
                '配置名称': summary['config_label'],
                '拓扑权重': summary['weights']['topology'],
                '图论权重': summary['weights']['graph'],
                '几何权重': summary['weights']['geometric'],
                '识别为2D数量': summary['num_candidates'],
                '2D比例': f"{summary['candidate_rate']:.1%}"
            })

        df_comparison = pd.DataFrame(comparison_data)
        comparison_file = self.output_dir / "config_comparison.xlsx"
        df_comparison.to_excel(comparison_file, index=False)

        print(f"✓ 配置对比表已生成: {comparison_file}")

        # 保存完整结果
        results_file = self.output_dir / "ablation_study_results.json"
        # 简化结果（移除详细的results避免文件过大）
        simplified_results = {}
        for config_name, summary in all_results.items():
            simplified_results[config_name] = {
                'config_label': summary['config_label'],
                'weights': summary['weights'],
                'total_processed': summary['total_processed'],
                'num_candidates': summary['num_candidates'],
                'candidate_rate': summary['candidate_rate'],
                'candidate_files': summary['candidate_files']
            }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, indent=2, ensure_ascii=False)

        print(f"✓ 完整结果已保存: {results_file}")

    def calculate_accuracy_from_evaluation(self, evaluation_file: str) -> None:
        """从人工评估结果计算准确率"""
        print(f"\n{'='*60}")
        print("计算各配置准确率")
        print(f"{'='*60}")

        # 读取人工评估结果
        df = pd.read_excel(evaluation_file)

        # 按配置分组统计
        accuracy_results = []

        for config_label in df['config_label'].unique():
            config_df = df[df['config_label'] == config_label]

            # 计算准确率（is_true_2d == 'yes'的比例）
            total = len(config_df)
            true_2d = len(config_df[config_df['is_true_2d'].str.lower() == 'yes'])
            accuracy = true_2d / total if total > 0 else 0

            # 计算高置信度准确率
            high_conf_df = config_df[config_df['confidence'].str.lower() == 'high']
            high_conf_accuracy = len(high_conf_df[high_conf_df['is_true_2d'].str.lower() == 'yes']) / len(high_conf_df) if len(high_conf_df) > 0 else 0

            accuracy_results.append({
                '配置名称': config_label,
                '识别数量': total,
                '真实2D数量': true_2d,
                '准确率': f"{accuracy:.1%}",
                '高置信度准确率': f"{high_conf_accuracy:.1%}"
            })

        # 保存准确率结果
        df_accuracy = pd.DataFrame(accuracy_results)
        accuracy_file = self.output_dir / "accuracy_analysis.xlsx"
        df_accuracy.to_excel(accuracy_file, index=False)

        print(f"✓ 准确率分析已生成: {accuracy_file}")
        print(f"\n准确率统计:")
        print(df_accuracy.to_string(index=False))

        # 生成推荐权重
        print(f"\n{'='*60}")
        print("基于准确率的权重推荐")
        print(f"{'='*60}")
        print("根据上述准确率结果，建议:")
        print("1. 选择准确率最高的配置作为主要依据")
        print("2. 或者将准确率作为权重，进行加权组合")
        print("3. 在论文中引用此消融实验结果作为权重选择的依据")


def main():
    """主函数"""
    # 配置参数
    input_dir = "data/filtered"  # 输入CIF文件目录
    output_dir = "data/ablation_study"  # 输出目录
    sample_size = 50  # 样本数量（可根据需要调整）

    # 创建并运行实验
    runner = AblationStudyRunner(input_dir, output_dir, sample_size)

    # 运行所有配置
    all_results = runner.run_all_configs()

    # 生成评估报告
    runner.generate_evaluation_report(all_results)

    print(f"\n{'='*60}")
    print("消融实验完成！")
    print(f"{'='*60}")
    print("\n后续步骤:")
    print("1. 打开 data/ablation_study/human_evaluation_template.xlsx")
    print("2. 人工评估每个候选材料是否为真实的二维材料")
    print("3. 填写 is_true_2d, confidence, notes 列")
    print("4. 保存为 human_evaluation_completed.xlsx")
    print("5. 运行以下命令计算准确率:")
    print("   python run_ablation_study.py --calculate-accuracy data/ablation_study/human_evaluation_completed.xlsx")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--calculate-accuracy':
        # 计算准确率模式
        evaluation_file = sys.argv[2] if len(sys.argv) > 2 else "data/ablation_study/human_evaluation_completed.xlsx"
        runner = AblationStudyRunner("data/filtered", "data/ablation_study")
        runner.calculate_accuracy_from_evaluation(evaluation_file)
    else:
        # 运行消融实验模式
        main()
