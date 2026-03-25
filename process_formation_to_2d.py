#!/usr/bin/env python3
"""
筛选材料二维化处理脚本
对mattergen_output/filter_Formation/下的材料进行二维化处理
输出到mattergen_output/filter_Formation_2D/

基于降为二维论文.pdf的方法实现
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加src路径
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "src" / "data"))

try:
    from src.data.crystal_2d_converter import Crystal2DConverter
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifParser, CifWriter
    PYMATGEN_AVAILABLE = True
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保安装了pymatgen: pip install pymatgen")
    PYMATGEN_AVAILABLE = False
    sys.exit(1)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FormationMaterials2DProcessor:
    """
    筛选材料二维化处理器
    专门处理filter_Formation目录下通过三个约束的高质量材料
    """
    
    def __init__(self, 
                 input_dir: str = "mattergen_output/filter_Formation",
                 output_dir: str = "mattergen_output/filter_Formation_2D"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化2D转换器（使用优化参数）
        self.converter = Crystal2DConverter(
            min_vacuum_thickness=15.0,    # 真空层厚度
            max_slab_thickness=20.0,      # 最大slab厚度
            min_atoms_2d=2,               # 最少原子数
            max_atoms_2d=200              # 最多原子数
        )
        
        # 结果统计
        self.stats = {
            'total_materials': 0,
            'successfully_converted': 0,
            'conversion_failed': 0,
            'total_2d_structures': 0,
            'conversion_methods_used': {},
            'material_details': []
        }
    
    def load_formation_materials(self) -> pd.DataFrame:
        """加载通过Formation Energy筛选的材料"""
        csv_file = self.input_dir / "final_materials.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"未找到材料文件: {csv_file}")
        
        df = pd.read_csv(csv_file)
        logger.info(f"加载了 {len(df)} 个通过Formation Energy筛选的材料")
        return df
    
    def process_single_material(self, material_info: Dict) -> Dict:
        """
        处理单个材料的二维化
        
        Args:
            material_info: 材料信息字典
            
        Returns:
            处理结果字典
        """
        material_id = material_info['material_id']
        formula = material_info['formula']
        cif_file = material_info['cif_file']
        
        logger.info(f"开始处理材料: {formula} ({material_id})")
        
        result = {
            'material_id': material_id,
            'formula': formula,
            'original_cif': cif_file,
            'conversion_success': False,
            'n_2d_structures': 0,
            '2d_cif_files': [],
            'conversion_methods': [],
            'error_message': None
        }
        
        try:
            # 检查CIF文件是否存在
            if not Path(cif_file).exists():
                # 尝试在input_dir中查找
                cif_path = self.input_dir / Path(cif_file).name
                if not cif_path.exists():
                    raise FileNotFoundError(f"CIF文件不存在: {cif_file}")
                cif_file = str(cif_path)
            
            # 读取结构
            parser = CifParser(cif_file)
            structures = parser.get_structures()
            
            if not structures:
                raise ValueError("无法从CIF文件读取结构")
            
            original_structure = structures[0]
            logger.info(f"  原始结构: {original_structure.composition.reduced_formula}")
            logger.info(f"  晶格参数: a={original_structure.lattice.a:.2f}, "
                       f"b={original_structure.lattice.b:.2f}, c={original_structure.lattice.c:.2f}")
            
            # 执行二维化转换
            generated_2d_structures = self.converter.generate_2d_structures_from_3d(original_structure)
            
            if generated_2d_structures:
                # 保存生成的2D结构
                saved_files = []
                methods_used = []
                
                for i, struct_2d in enumerate(generated_2d_structures):
                    # 构造输出文件名
                    conversion_method = struct_2d.properties.get('conversion_method', 'unknown')
                    output_filename = f"{material_id}_2d_{conversion_method}_{i}.cif"
                    output_filepath = self.output_dir / output_filename
                    
                    # 保存CIF文件
                    writer = CifWriter(struct_2d)
                    writer.write_file(str(output_filepath))
                    
                    saved_files.append(str(output_filepath))
                    methods_used.append(conversion_method)
                    
                    # 记录2D结构信息
                    logger.info(f"    生成2D结构 {i+1}: {conversion_method}方法")
                    logger.info(f"      原子数: {len(struct_2d.sites)}")
                    logger.info(f"      真空层厚度: {struct_2d.lattice.c:.2f}Å")
                
                result.update({
                    'conversion_success': True,
                    'n_2d_structures': len(generated_2d_structures),
                    '2d_cif_files': saved_files,
                    'conversion_methods': list(set(methods_used))
                })
                
                # 更新方法统计
                for method in set(methods_used):
                    self.stats['conversion_methods_used'][method] = \
                        self.stats['conversion_methods_used'].get(method, 0) + 1
                
                logger.info(f"  ✅ 成功生成 {len(generated_2d_structures)} 个2D结构")
                
            else:
                result['error_message'] = "未能生成有效的2D结构"
                logger.warning(f"  ❌ 未能为 {formula} 生成2D结构")
            
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            result['error_message'] = error_msg
            logger.error(f"  ❌ {formula} 处理失败: {e}")
        
        return result
    
    def process_all_materials(self) -> Dict:
        """处理所有筛选出的材料"""
        logger.info("=== 开始批量二维化处理 ===")
        
        # 加载材料列表
        materials_df = self.load_formation_materials()
        self.stats['total_materials'] = len(materials_df)
        
        # 逐个处理材料
        for idx, row in materials_df.iterrows():
            material_info = row.to_dict()
            result = self.process_single_material(material_info)
            
            # 更新统计
            if result['conversion_success']:
                self.stats['successfully_converted'] += 1
                self.stats['total_2d_structures'] += result['n_2d_structures']
            else:
                self.stats['conversion_failed'] += 1
            
            self.stats['material_details'].append(result)
            
            logger.info(f"进度: {idx+1}/{len(materials_df)}")
        
        # 生成处理结果
        self.generate_results_summary()
        
        return self.stats
    
    def generate_results_summary(self):
        """生成处理结果摘要"""
        # 保存详细结果
        results_df = pd.DataFrame(self.stats['material_details'])
        results_csv = self.output_dir / "2d_conversion_results.csv"
        results_df.to_csv(results_csv, index=False)
        
        # 创建成功转换的材料列表
        successful_materials = results_df[results_df['conversion_success'] == True]
        if len(successful_materials) > 0:
            success_csv = self.output_dir / "successful_2d_materials.csv"
            successful_materials.to_csv(success_csv, index=False)
            
            # 创建最终的2D材料目录清单
            all_2d_files = []
            for _, row in successful_materials.iterrows():
                for cif_file in row['2d_cif_files']:
                    all_2d_files.append({
                        'original_material_id': row['material_id'],
                        'original_formula': row['formula'],
                        '2d_cif_file': cif_file,
                        'conversion_method': Path(cif_file).stem.split('_2d_')[1].rsplit('_', 1)[0],
                        'file_exists': Path(cif_file).exists()
                    })
            
            catalog_df = pd.DataFrame(all_2d_files)
            catalog_csv = self.output_dir / "2d_materials_catalog.csv"
            catalog_df.to_csv(catalog_csv, index=False)
        
        # 保存统计报告
        report = {
            "processing_summary": {
                "total_input_materials": self.stats['total_materials'],
                "successfully_converted": self.stats['successfully_converted'],
                "conversion_failed": self.stats['conversion_failed'],
                "success_rate": f"{self.stats['successfully_converted']/self.stats['total_materials']*100:.1f}%" if self.stats['total_materials'] > 0 else "0%",
                "total_2d_structures_generated": self.stats['total_2d_structures']
            },
            "conversion_methods_statistics": self.stats['conversion_methods_used'],
            "failed_materials": [
                {
                    "material_id": detail['material_id'],
                    "formula": detail['formula'],
                    "error": detail['error_message']
                }
                for detail in self.stats['material_details']
                if not detail['conversion_success']
            ]
        }
        
        report_file = self.output_dir / "2d_processing_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印结果摘要
        logger.info("\n=== 二维化处理完成 ===")
        logger.info(f"输入材料总数: {self.stats['total_materials']}")
        logger.info(f"成功转换: {self.stats['successfully_converted']}")
        logger.info(f"转换失败: {self.stats['conversion_failed']}")
        logger.info(f"成功率: {self.stats['successfully_converted']/self.stats['total_materials']*100:.1f}%")
        logger.info(f"生成2D结构总数: {self.stats['total_2d_structures']}")
        
        if self.stats['conversion_methods_used']:
            logger.info(f"\n转换方法统计:")
            for method, count in self.stats['conversion_methods_used'].items():
                logger.info(f"  {method}: {count}个结构")
        
        logger.info(f"\n输出文件:")
        logger.info(f"  详细结果: {results_csv}")
        logger.info(f"  成功材料: {self.output_dir}/successful_2d_materials.csv")
        logger.info(f"  2D材料目录: {self.output_dir}/2d_materials_catalog.csv")
        logger.info(f"  处理报告: {report_file}")
        logger.info(f"  2D结构文件: {self.output_dir}/*.cif")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="对筛选材料进行二维化处理")
    parser.add_argument("--input-dir", type=str, 
                       default="mattergen_output/filter_Formation",
                       help="输入目录（默认: mattergen_output/filter_Formation）")
    parser.add_argument("--output-dir", type=str,
                       default="mattergen_output/filter_Formation_2D", 
                       help="输出目录（默认: mattergen_output/filter_Formation_2D）")
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not Path(args.input_dir).exists():
        logger.error(f"输入目录不存在: {args.input_dir}")
        return 1
    
    # 检查必需文件
    csv_file = Path(args.input_dir) / "final_materials.csv"
    if not csv_file.exists():
        logger.error(f"未找到材料文件: {csv_file}")
        logger.error("请先运行Formation Energy筛选脚本")
        return 1
    
    try:
        # 创建处理器并执行
        processor = FormationMaterials2DProcessor(args.input_dir, args.output_dir)
        results = processor.process_all_materials()
        
        if results['successfully_converted'] > 0:
            logger.info(f"\n🎉 成功! 共转换了 {results['successfully_converted']} 个材料")
            logger.info(f"生成了 {results['total_2d_structures']} 个2D结构")
            logger.info(f"结果保存在: {args.output_dir}")
        else:
            logger.warning("⚠️ 没有材料成功转换为2D结构")
            logger.info("建议检查输入材料的结构特征")
        
        return 0
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)