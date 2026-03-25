# -*- coding: utf-8 -*-
"""
简化材料筛选脚本

由于能量数据不可用（都为0），只基于合成性约束进行筛选：
1. 元素种类 ≤ 3

"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure, Element
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
import shutil

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleMaterialFilter:
    """简化材料筛选器（仅基于元素种类）"""
    
    def __init__(self, raw_dir="data/raw", output_dir="data/filtered"):
        """
        初始化筛选器
        
        Args:
            raw_dir: 原始数据目录
            output_dir: 输出目录
        """
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 筛选判据
        self.criteria = {
            'max_elements': 3,  # 合成性：元素种类 ≤ 3
        }
        
        # 统计数据
        self.stats = {
            'total_files': 0,
            'parsed_successfully': 0,
            'passed_element_filter': 0,
            'is_2d_count': 0,
            'final_selected': 0,
            'failed_parsing': [],
            'element_distribution': {},
            'dimensionality_distribution': {}
        }
    
    def check_material(self, structure):
        """
        检查材料是否满足元素种类条件
        
        Args:
            structure: pymatgen Structure对象
            
        Returns:
            tuple: (is_valid, reasons, info)
        """
        reasons = []
        info = {}
        
        try:
            # 检查元素数量 ≤ 3
            n_elements = len(structure.composition.elements)
            elements = [str(el) for el in structure.composition.elements]
            
            info['n_elements'] = n_elements
            info['elements'] = elements
            
            # 统计元素分布
            self.stats['element_distribution'][n_elements] = self.stats['element_distribution'].get(n_elements, 0) + 1
            
            if n_elements > self.criteria['max_elements']:
                reasons.append(f"元素数过多: {n_elements} > {self.criteria['max_elements']}")
                return False, reasons, info
            else:
                self.stats['passed_element_filter'] += 1
            
            # 检查维度信息（仅用于统计，不筛选）
            try:
                dimensionality = get_dimensionality_larsen(structure)
                info['dimensionality'] = dimensionality
                info['is_2d'] = (dimensionality == 2)
                
                # 统计维度分布
                self.stats['dimensionality_distribution'][dimensionality] = self.stats['dimensionality_distribution'].get(dimensionality, 0) + 1
                
                if dimensionality == 2:
                    self.stats['is_2d_count'] += 1
                    
            except Exception as e:
                info['dimensionality'] = None
                info['is_2d'] = False
                logger.debug(f"维度分析失败: {e}")
            
            return True, [], info
            
        except Exception as e:
            reasons.append(f"材料检查出错: {e}")
            return False, reasons, info
    
    def get_material_info(self, structure, cif_path):
        """
        提取材料的基本信息
        
        Args:
            structure: pymatgen Structure对象
            cif_path: CIF文件路径
            
        Returns:
            dict: 材料信息字典
        """
        try:
            # 从文件名提取Material ID
            material_id = Path(cif_path).stem
            
            # 基本信息
            info = {
                'material_id': material_id,
                'formula': structure.composition.reduced_formula,
                'n_atoms': len(structure.sites),
                'density': structure.density,
                'volume': structure.volume,
                'lattice_params': {
                    'a': structure.lattice.a,
                    'b': structure.lattice.b,
                    'c': structure.lattice.c,
                    'alpha': structure.lattice.alpha,
                    'beta': structure.lattice.beta,
                    'gamma': structure.lattice.gamma
                }
            }
            
            # 空间群信息
            try:
                sga = SpacegroupAnalyzer(structure)
                info['space_group'] = sga.get_space_group_number()
                info['space_group_symbol'] = sga.get_space_group_symbol()
            except:
                info['space_group'] = None
                info['space_group_symbol'] = None
            
            return info
            
        except Exception as e:
            logger.error(f"提取材料信息失败 {cif_path}: {e}")
            return {}
    
    def process_single_file(self, cif_path):
        """
        处理单个CIF文件
        
        Args:
            cif_path: CIF文件路径
            
        Returns:
            tuple: (success, material_info, messages)
        """
        messages = []
        
        try:
            # 解析CIF文件
            parser = CifParser(str(cif_path))
            structures = parser.parse_structures()
            
            if not structures:
                messages.append("无法解析结构")
                return False, {}, messages
            
            structure = structures[0]  # 取第一个结构
            
            # 获取材料信息
            material_info = self.get_material_info(structure, cif_path)
            
            # 检查是否满足条件
            is_valid, check_reasons, check_info = self.check_material(structure)
            
            # 合并信息
            material_info.update(check_info)
            material_info['check_result'] = {
                'passed': is_valid,
                'reasons': check_reasons
            }
            
            if not is_valid:
                messages.extend([f"筛选失败: {r}" for r in check_reasons])
                return False, material_info, messages
            
            messages.append("通过筛选检查")
            return True, material_info, messages
            
        except Exception as e:
            messages.append(f"处理文件出错: {e}")
            self.stats['failed_parsing'].append(str(cif_path))
            return False, {}, messages
    
    def filter_materials(self, max_files=None):
        """
        筛选材料
        
        Args:
            max_files: 最大处理文件数，None表示处理所有文件
        """
        logger.info("开始基于元素种类筛选材料...")
        logger.info(f"原始数据目录: {self.raw_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"筛选条件: 元素种类 ≤ {self.criteria['max_elements']}")
        
        # 获取所有CIF文件
        cif_files = list(self.raw_dir.glob("*.cif"))
        
        if not cif_files:
            logger.error(f"在{self.raw_dir}中未找到CIF文件")
            return
        
        if max_files:
            cif_files = cif_files[:max_files]
        
        self.stats['total_files'] = len(cif_files)
        logger.info(f"找到{len(cif_files)}个CIF文件")
        
        # 存储所有材料信息
        all_materials_info = []
        selected_materials = []
        
        # 处理每个文件
        for cif_path in tqdm(cif_files, desc="筛选材料"):
            success, material_info, messages = self.process_single_file(cif_path)
            
            if success:
                # 复制文件到输出目录
                output_path = self.output_dir / cif_path.name
                try:
                    shutil.copy2(cif_path, output_path)
                    material_info['output_path'] = str(output_path)
                    selected_materials.append(material_info)
                    self.stats['final_selected'] += 1
                except Exception as e:
                    logger.error(f"复制文件失败 {cif_path}: {e}")
            else:
                self.stats['parsed_successfully'] += 1 if material_info else 0
            
            # 保存材料信息（无论是否通过筛选）
            if material_info:
                material_info['selected'] = success
                material_info['messages'] = messages
                all_materials_info.append(material_info)
        
        # 保存筛选结果
        self.save_results(all_materials_info, selected_materials)
        
        # 打印统计信息
        self.print_statistics()
    
    def save_results(self, all_materials_info, selected_materials):
        """保存筛选结果"""
        
        # 保存所有材料信息
        all_info_path = self.output_dir / "all_materials_info.json"
        with open(all_info_path, 'w', encoding='utf-8') as f:
            json.dump(all_materials_info, f, ensure_ascii=False, indent=2)
        
        # 保存筛选通过的材料信息
        selected_info_path = self.output_dir / "selected_materials_info.json"
        with open(selected_info_path, 'w', encoding='utf-8') as f:
            json.dump(selected_materials, f, ensure_ascii=False, indent=2)
        
        # 保存为CSV格式（便于分析）
        if selected_materials:
            df_data = []
            for mat in selected_materials:
                row = {
                    'material_id': mat.get('material_id', ''),
                    'formula': mat.get('formula', ''),
                    'n_atoms': mat.get('n_atoms', 0),
                    'n_elements': mat.get('n_elements', 0),
                    'elements': ', '.join(mat.get('elements', [])),
                    'density': mat.get('density', 0),
                    'volume': mat.get('volume', 0),
                    'space_group': mat.get('space_group', ''),
                    'dimensionality': mat.get('dimensionality', ''),
                    'is_2d': mat.get('is_2d', False),
                    'lattice_a': mat.get('lattice_params', {}).get('a', 0),
                    'lattice_b': mat.get('lattice_params', {}).get('b', 0),
                    'lattice_c': mat.get('lattice_params', {}).get('c', 0),
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            csv_path = self.output_dir / "selected_materials.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"筛选结果已保存到 {csv_path}")
            
            # 单独保存2D材料
            df_2d = df[df['is_2d'] == True]
            if len(df_2d) > 0:
                csv_2d_path = self.output_dir / "selected_2d_materials.csv"
                df_2d.to_csv(csv_2d_path, index=False, encoding='utf-8')
                logger.info(f"2D材料筛选结果已保存到 {csv_2d_path}")
        
        # 保存统计信息
        stats_path = self.output_dir / "filtering_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"筛选结果已保存到 {self.output_dir}")
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("材料筛选统计结果（基于元素种类）")
        print("="*60)
        print(f"总文件数: {self.stats['total_files']}")
        print(f"成功解析: {self.stats['parsed_successfully']}")
        print(f"通过元素筛选: {self.stats['passed_element_filter']}")
        print(f"最终筛选通过: {self.stats['final_selected']}")
        print(f"其中2D材料: {self.stats['is_2d_count']}")
        
        if self.stats['total_files'] > 0:
            success_rate = self.stats['final_selected'] / self.stats['total_files'] * 100
            print(f"筛选通过率: {success_rate:.2f}%")
            
            if self.stats['final_selected'] > 0:
                twod_rate = self.stats['is_2d_count'] / self.stats['final_selected'] * 100
                print(f"2D材料比例: {twod_rate:.2f}%")
        
        print(f"\n解析失败文件数: {len(self.stats['failed_parsing'])}")
        
        print(f"\n元素数量分布:")
        for n_elements, count in sorted(self.stats['element_distribution'].items()):
            percentage = count / self.stats['total_files'] * 100
            print(f"  {n_elements}种元素: {count} ({percentage:.1f}%)")
        
        if self.stats['dimensionality_distribution']:
            print(f"\n维度分布:")
            for dim, count in sorted(self.stats['dimensionality_distribution'].items()):
                percentage = count / self.stats['total_files'] * 100
                print(f"  {dim}D材料: {count} ({percentage:.1f}%)")
        
        print("="*60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="基于元素种类筛选材料")
    parser.add_argument("--raw_dir", type=str, default="data/raw",
                       help="原始数据目录")
    parser.add_argument("--output_dir", type=str, default="data/filtered", 
                       help="输出目录")
    parser.add_argument("--max_files", type=int, default=None,
                       help="最大处理文件数")
    parser.add_argument("--max_elements", type=int, default=3,
                       help="最大元素数")
    
    args = parser.parse_args()
    
    # 创建筛选器
    filter_obj = SimpleMaterialFilter(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir
    )
    
    # 更新筛选条件
    if args.max_elements:
        filter_obj.criteria['max_elements'] = args.max_elements
    
    # 执行筛选
    filter_obj.filter_materials(max_files=args.max_files)


if __name__ == "__main__":
    main()