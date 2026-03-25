"""
纯MatterGen材料生成脚本
仅使用MatterGen生成材料，不包含模拟功能
"""

import argparse
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from generate_stable_materials import MaterialGenerator


def resolve_checkpoint_path(checkpoint_input: str) -> str:
    """解析checkpoint路径"""
    checkpoint_path = Path(checkpoint_input)
    
    # 情况1: 绝对路径，直接检查
    if checkpoint_path.is_absolute():
        if checkpoint_path.exists():
            return str(checkpoint_path)
        return None
    
    # 情况2: 相对路径，检查当前目录
    if checkpoint_path.exists():
        return str(checkpoint_path.absolute())
    
    # 情况3: 预训练模型名称
    mattergen_base = Path("C:/Users/Administrator/Downloads/mattergen-main/checkpoints")
    if mattergen_base.exists():
        # 直接匹配模型名称
        model_path = mattergen_base / checkpoint_input
        if model_path.exists() and (model_path / "config.yaml").exists():
            return str(model_path)
        
        # 模糊匹配
        for model_dir in mattergen_base.iterdir():
            if model_dir.is_dir() and checkpoint_input.lower() in model_dir.name.lower():
                if (model_dir / "config.yaml").exists():
                    return str(model_dir)
    
    return None


def show_available_models():
    """显示可用的预训练模型"""
    mattergen_base = Path("C:/Users/Administrator/Downloads/mattergen-main/checkpoints")
    if not mattergen_base.exists():
        print("  未找到MatterGen预训练模型目录")
        return
    
    models = []
    for model_dir in mattergen_base.iterdir():
        if model_dir.is_dir() and (model_dir / "config.yaml").exists():
            models.append(model_dir.name)
    
    if models:
        for model in sorted(models):
            print(f"  - {model}")
        print(f"\n使用示例:")
        print(f"  python run_mattergen_generation.py --checkpoint {models[0]} --num-samples 10")
    else:
        print("  未找到可用的预训练模型")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用MatterGen生成稳定材料")
    parser.add_argument("--checkpoint", type=str, default="mp_20_base",
                       help="MatterGen模型checkpoint路径 (默认: mp_20_base)")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="生成材料数量")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="批处理大小")
    parser.add_argument("--output-dir", type=str, default="mattergen_output",
                       help="输出目录")
    
    # 约束条件参数
    parser.add_argument("--target-compositions", type=str, default=None,
                       help="目标组成约束 (JSON格式), 例: '[{\"Mo\": 1, \"S\": 2}]'")
    parser.add_argument("--max-elements", type=int, default=3,
                       help="最大元素种类数 (默认: 3)")
    parser.add_argument("--elements", type=str, default=None,
                       help="允许的元素列表 (逗号分隔), 例: 'Mo,W,S,Se,Te'")
    parser.add_argument("--energy-above-hull", type=float, default=None,
                       help="最大hull能量约束 (eV/atom)")
    parser.add_argument("--formation-energy", type=float, default=None,
                       help="最大形成能约束 (eV/atom)")
    parser.add_argument("--band-gap", type=float, default=None,
                       help="目标带隙约束 (eV)")
    parser.add_argument("--force-2d", action="store_true",
                       help="强制生成二维材料特征")
    
    args = parser.parse_args()
    
    print("=== MatterGen材料生成程序 ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"生成数量: {args.num_samples}")
    print(f"批处理大小: {args.batch_size}")
    print(f"输出目录: {args.output_dir}")
    
    # 显示约束条件
    constraints = []
    if args.target_compositions:
        constraints.append(f"目标组成: {args.target_compositions}")
    if args.max_elements != 3:
        constraints.append(f"最大元素数: {args.max_elements}")
    if args.elements:
        constraints.append(f"允许元素: {args.elements}")
    if args.energy_above_hull is not None:
        constraints.append(f"Hull能量: <{args.energy_above_hull} eV/atom")
    if args.formation_energy is not None:
        constraints.append(f"形成能: <{args.formation_energy} eV/atom")
    if args.band_gap is not None:
        constraints.append(f"带隙: ~{args.band_gap} eV")
    if args.force_2d:
        constraints.append("强制二维材料")
    
    if constraints:
        print("约束条件:")
        for constraint in constraints:
            print(f"  - {constraint}")
    else:
        print("约束条件: 无约束生成")
    
    # 检查和解析checkpoint路径
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    if not checkpoint_path:
        print(f"错误: 无法找到有效的模型: {args.checkpoint}")
        print("\n可用的预训练模型:")
        show_available_models()
        return 1
    
    print(f"解析后的模型路径: {checkpoint_path}")
    
    try:
        # 创建生成器
        generator = MaterialGenerator(checkpoint_path, args.output_dir)
        
        # 构建约束参数
        constraints = {
            'target_compositions': args.target_compositions,
            'max_elements': args.max_elements,
            'elements': args.elements.split(',') if args.elements else None,
            'energy_above_hull': args.energy_above_hull,
            'formation_energy': args.formation_energy,
            'band_gap': args.band_gap,
            'force_2d': args.force_2d
        }
        
        print("\n--- 步骤1: 使用MatterGen生成材料 ---")
        cif_files = generator.generate_materials_mattergen(
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            constraints=constraints
        )
        
        if not cif_files:
            print("错误: 没有生成任何材料文件")
            return 1
        
        print(f"成功生成 {len(cif_files)} 个材料文件")
        
        print("\n--- 步骤2: 分析材料结构 ---")
        try:
            original_df = generator.analyze_materials(cif_files)
            print(f"成功分析 {len(original_df)} 个材料")
        except ImportError as e:
            print(f"分析失败: {e}")
            print("跳过分析步骤，仅保留生成的CIF文件")
            print_files_summary(cif_files, args.output_dir)
            return 0
        
        print("\n--- 步骤3: 应用筛选标准 ---")
        filtered_df = generator.filter_materials(original_df)
        
        print("\n--- 步骤4: 生成报告 ---")
        generator.generate_report(original_df, filtered_df)
        
        print("\n=== 生成完成! ===")
        print_results_summary(original_df, filtered_df, args.output_dir)
        
        return 0
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


def print_files_summary(cif_files, output_dir):
    """打印文件摘要"""
    print(f"\n生成的文件:")
    print(f"   输出目录: {output_dir}")
    print(f"   CIF文件数量: {len(cif_files)}")
    print(f"   CIF文件位置: {output_dir}/cif_files/")
    
    # 显示前几个文件名
    print(f"\n文件列表（前10个）:")
    for i, cif_file in enumerate(cif_files[:10], 1):
        filename = Path(cif_file).name
        print(f"   {i}. {filename}")
    
    if len(cif_files) > 10:
        print(f"   ... 还有 {len(cif_files) - 10} 个文件")


def print_results_summary(original_df, filtered_df, output_dir):
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
        print(f"\n推荐材料（前5个）:")
        for i, material in enumerate(filtered_df.head(5).itertuples(), 1):
            score = getattr(material, 'overall_score', 0)
            formula = getattr(material, 'formula', 'Unknown')
            print(f"   {i}. {formula} (综合评分: {score:.3f})")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)