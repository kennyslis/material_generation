"""
运行二维材料二次筛选

使用方法：
python run_2d_screening.py

该脚本将：
1. 从 data/filtered 读取已筛选的合成性材料（<=3元素）
2. 应用二维材料识别算法
3. 输出真正的二维材料候选到 data/2d_materials
"""

import sys
import os
from pathlib import Path

# 添加src路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data.filter_2d_materials_secondary import TwoDMaterialScreener

def main():
    print("=" * 60)
    print("二维材料二次筛选 - 基于论文方法")
    print("=" * 60)
    
    # 检查输入目录
    input_dir = Path("data/filtered")
    if not input_dir.exists():
        print(f"错误：输入目录不存在 {input_dir}")
        print("请先运行合成性筛选脚本生成筛选数据")
        return
    
    cif_count = len(list(input_dir.glob("*.cif")))
    if cif_count == 0:
        print(f"错误：在 {input_dir} 中未找到CIF文件")
        return
    
    print(f"输入目录: {input_dir}")
    print(f"找到 {cif_count} 个预筛选材料")
    
    # 创建输出目录
    output_dir = Path("data/2d_materials")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    print()
    
    # 创建筛选器并运行
    screener = TwoDMaterialScreener(str(input_dir), str(output_dir))
    
    # 配置筛选参数
    print("筛选参数:")
    print(f"- 最大元素种类: {screener.max_elements}")
    print(f"- 最大层间结合能: {screener.max_binding_energy} eV/A^2")
    print(f"- 最小层间距离: {screener.min_interlayer_distance} A")
    print()
    
    try:
        screener.run_screening()
        
        print("\n" + "=" * 60)
        print("筛选完成！")
        print("=" * 60)
        
        # 显示结果文件
        results_files = [
            output_dir / "2d_screening_results.json",
            output_dir / "2d_material_candidates.json", 
            output_dir / "2d_screening_summary.csv"
        ]
        
        print("\n生成的文件:")
        for file_path in results_files:
            if file_path.exists():
                print(f"✓ {file_path}")
            else:
                print(f"✗ {file_path}")
        
        # 读取并显示候选材料概述
        candidates_file = output_dir / "2d_material_candidates.json"
        if candidates_file.exists():
            import json
            with open(candidates_file, 'r', encoding='utf-8') as f:
                candidates = json.load(f)
            
            print(f"\n发现 {len(candidates)} 个二维材料候选:")
            for i, candidate in enumerate(candidates[:10], 1):  # 显示前10个
                formula = candidate['formula']
                score = candidate['score']
                dimension = candidate['tsa_analysis']['dimension']
                print(f"{i:2d}. {formula:15s} (评分: {score:.3f}, TSA: {dimension})")
            
            if len(candidates) > 10:
                print(f"... 以及其他 {len(candidates) - 10} 个候选材料")
        
    except Exception as e:
        print(f"筛选过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()