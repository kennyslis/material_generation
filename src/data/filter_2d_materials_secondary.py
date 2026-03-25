"""
二维材料二次筛选脚本
基于论文"二维材料的高通量筛选与光催化性能预测"中的方法

筛选流程：
1. 读取已经按合成性约束筛选的材料（<=3种元素）
2. 应用二维材料识别算法
3. 计算层间结合能和剥离能
4. 输出真正的二维材料候选

主要算法：
- 拓扑缩放算法(TSA)
- 层间结合能计算
- vdW相互作用分析
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
# from pymatgen.analysis.dimensionality import get_dimensionality_larsen  # 可能不存在，使用备用方法
try:
    from pymatgen.analysis.graphs import StructureGraph
except ImportError:
    StructureGraph = None
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pandas as pd
from ase.io import read
from ase import Atoms
import matplotlib.pyplot as plt
try:
    from scipy.spatial.distance import pdist, squareform
except ImportError:
    pdist = None
    squareform = None

class TwoDMaterialScreener:
    """二维材料二次筛选器"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 筛选参数
        self.max_elements = 3  # 最大元素种类
        self.max_binding_energy = 0.15  # 最大层间结合能 (eV/Å²)
        self.min_interlayer_distance = 2.5  # 最小层间距离 (Å)
        self.min_2d_score = 0.3  # 降低最小评分阈值
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'dimensionality_2d': 0,
            'low_binding_energy': 0,
            'layered_structure': 0,
            'final_2d_candidates': 0,
            'processing_errors': 0
        }
        
    def topology_scaling_analysis(self, structure: Structure) -> Dict:
        """
        简化的拓扑缩放算法(TSA)分析材料维度
        基于晶格参数和原子分布的快速分析
        """
        try:
            # 使用几何特征进行快速维度判断
            lattice = structure.lattice
            a, b, c = lattice.abc
            
            # 计算轴比
            ratios = [c/a, c/b, a/b, b/a, a/c, b/c]
            max_ratio = max(ratios)
            min_ratio = min(ratios)
            
            # 分析原子在不同方向的分布
            coords = np.array([site.frac_coords for site in structure])
            
            # 计算各方向的原子分布范围
            coord_ranges = [
                coords[:, 0].max() - coords[:, 0].min(),  # a方向
                coords[:, 1].max() - coords[:, 1].min(),  # b方向
                coords[:, 2].max() - coords[:, 2].min()   # c方向
            ]
            
   
            # 如果某个方向的原子分布很小且轴比很大，可能是低维材料
            dimension = '3D'  # 默认3D
            scaling_ratio = 1.0
            confidence = 0.5
            
            # 2D判断：c轴显著大于a,b轴，且原子主要分布在xy平面
            if (c/a > 2.0 and c/b > 2.0) or (coord_ranges[2] < 0.3 and max_ratio > 2.0):
                dimension = '2D'
                scaling_ratio = 2.0
                confidence = 0.8
            
            # 1D判断：有两个轴显著小于第三个轴
            elif max_ratio > 5.0 and min_ratio < 0.5:
                dimension = '1D' 
                scaling_ratio = 4.0
                confidence = 0.7
            
            # 0D判断：所有轴都很相似但结构很紧凑
            elif max_ratio < 1.5 and len(structure) < 10:
                dimension = '0D'
                scaling_ratio = 8.0
                confidence = 0.6
            
            return {
                'dimension': dimension,
                'scaling_ratio': scaling_ratio,
                'confidence': confidence,
                'lattice_ratios': ratios,
                'coord_ranges': coord_ranges
            }
            
        except Exception as e:
            print(f"TSA分析错误: {e}")
            return {'dimension': 'Unknown', 'scaling_ratio': 0.0, 'confidence': 0.0}
    
    def _get_connected_clusters(self, structure_graph: StructureGraph) -> List:
        """获取连接的原子团簇"""
        try:
            # 使用图论方法找到连通分量
            import networkx as nx
            
            # 创建网络图
            G = nx.Graph()
            for i in range(len(structure_graph.structure)):
                G.add_node(i)
            
            # 尝试不同的API访问方式
            try:
                # 新API方式
                graph_dict = structure_graph.graph
                for i, connections in graph_dict.items():
                    for j in connections:
                        if i < j:  # 避免重复边
                            G.add_edge(i, j)
            except (AttributeError, TypeError):
                try:
                    # 备用方式：直接遍历图
                    for i in range(len(structure_graph.structure)):
                        neighbors = structure_graph.get_connected_sites(i)
                        for neighbor in neighbors:
                            j = neighbor.index
                            if i < j:
                                G.add_edge(i, j)
                except:
                    # 最简单的方式：假设所有原子独立
                    pass
            
            # 找到连通分量
            connected_components = list(nx.connected_components(G))
            return connected_components if connected_components else [set([i]) for i in range(len(structure_graph.structure))]
            
        except Exception:
            # 如果出错，返回简单的原子索引列表
            return [set([i]) for i in range(len(structure_graph.structure))]
    
    def _simple_dimensionality_analysis(self, structure: Structure) -> int:
        """维度分析方法"""
        try:
            # 基于晶格参数和原子分布的简单维度判断
            lattice = structure.lattice
            a, b, c = lattice.abc
            
            # 如果某个轴显著大于其他轴，可能是层状（2D）或链状（1D）
            ratios = [c/a, c/b, a/b, b/a, a/c, b/c]
            max_ratio = max(ratios)
            
            if max_ratio > 3.0:
                # 有一个轴显著长，可能是层状或链状
                return 2  # 假设为2D
            else:
                return 3  # 默认为3D
                
        except Exception:
            return 3  # 默认3D
    
    def analyze_layered_structure(self, structure: Structure) -> Dict:
        """分析层状结构特征"""
        try:
            # 使用简化的维度分析（备用方法）
            try:
                from pymatgen.analysis.dimensionality import get_dimensionality_larsen
                dimensionality = get_dimensionality_larsen(structure)
            except ImportError:
                # 如果模块不存在，使用简化方法
                dimensionality = self._simple_dimensionality_analysis(structure)
            
            # 分析晶格参数
            lattice = structure.lattice
            a, b, c = lattice.abc
            alpha, beta, gamma = lattice.angles
            
            # 计算层间距离（基于c轴方向的原子间距）
            interlayer_distances = self._calculate_interlayer_distances(structure)
            
            # 分析化学键网络
            bond_analysis = self._analyze_bonding_network(structure)
            
            return {
                'larsen_dimensionality': dimensionality,
                'lattice_ratios': {'c/a': c/a, 'c/b': c/b},
                'interlayer_distances': interlayer_distances,
                'avg_interlayer_distance': np.mean(interlayer_distances) if interlayer_distances else 0,
                'bond_analysis': bond_analysis,
                'is_layered': self._is_layered_structure(structure, interlayer_distances)
            }
            
        except Exception as e:
            print(f"层状结构分析错误: {e}")
            return {'larsen_dimensionality': 3, 'is_layered': False}
    
    def _calculate_interlayer_distances(self, structure: Structure) -> List[float]:
        """计算层间距离"""
        try:
            # 沿c轴方向寻找原子间的大间隙
            distances = []
            c_coords = [site.frac_coords[2] for site in structure]
            c_coords.sort()
            
            # 计算相邻原子层间的距离
            for i in range(len(c_coords) - 1):
                if c_coords[i+1] - c_coords[i] > 0.1:  # 只考虑显著间隙
                    distance = (c_coords[i+1] - c_coords[i]) * structure.lattice.c
                    distances.append(distance)
            
            return distances
            
        except Exception:
            return []
    
    def _analyze_bonding_network(self, structure: Structure) -> Dict:
        """简化的化学键网络分析"""
        try:
            # 使用简单的距离分析代替复杂的图分析
            coords = np.array([site.coords for site in structure])
            n_atoms = len(coords)
            
            if n_atoms < 2:
                return {'total_bonds': 0, 'xy_bond_ratio': 0}
            
            # 计算原子间距离
            if pdist is None or squareform is None:
                # 手动计算距离矩阵
                distances = np.zeros((n_atoms, n_atoms))
                for i in range(n_atoms):
                    for j in range(i+1, n_atoms):
                        dist = np.linalg.norm(coords[i] - coords[j])
                        distances[i, j] = distances[j, i] = dist
            else:
                distances = squareform(pdist(coords))
            
            # 简单的键判断：距离在合理范围内（1.0-4.0 Å）
            min_bond_dist = 1.0
            max_bond_dist = 4.0
            
            bonds_xy = 0
            bonds_z = 0
            total_bonds = 0
            
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    dist = distances[i, j]
                    if min_bond_dist < dist < max_bond_dist:
                        total_bonds += 1
                        
                        # 判断键的方向
                        z_diff = abs(coords[i][2] - coords[j][2])
                        xy_diff = np.sqrt((coords[i][0] - coords[j][0])**2 + 
                                        (coords[i][1] - coords[j][1])**2)
                        
                        if z_diff > xy_diff:
                            bonds_z += 1
                        else:
                            bonds_xy += 1
            
            xy_ratio = bonds_xy / total_bonds if total_bonds > 0 else 0
            
            return {
                'total_bonds': total_bonds,
                'xy_plane_bonds': bonds_xy,
                'z_direction_bonds': bonds_z,
                'xy_bond_ratio': xy_ratio
            }
            
        except Exception:
            return {'total_bonds': 0, 'xy_bond_ratio': 0}
    
    def _is_layered_structure(self, structure: Structure, interlayer_distances: List[float]) -> bool:
        """判断是否为层状结构"""
        try:
            # 条件1：存在显著的层间距离
            has_large_gap = any(d > self.min_interlayer_distance for d in interlayer_distances)
            
            # 条件2：c轴与a、b轴的比例（放宽条件）
            lattice = structure.lattice
            c_to_a_ratio = lattice.c / lattice.a
            c_to_b_ratio = lattice.c / lattice.b
            
            # 条件3：原子密度分布
            density_anisotropy = self._calculate_density_anisotropy(structure)
            
            # 放宽层状结构判断条件
            has_axis_ratio = (c_to_a_ratio > 1.2 or c_to_b_ratio > 1.2 or 
                            1/c_to_a_ratio > 1.2 or 1/c_to_b_ratio > 1.2)
            
            return (has_large_gap or has_axis_ratio) and density_anisotropy > 0.2
            
        except Exception:
            return False
    
    def _calculate_density_anisotropy(self, structure: Structure) -> float:
        """计算密度各向异性"""
        try:
            # 简单的各向异性度量：c方向vs xy平面的原子密度差异
            lattice = structure.lattice
            volume = lattice.volume
            natoms = len(structure)
            
            # xy平面密度
            xy_area = lattice.a * lattice.b * np.sin(np.radians(lattice.gamma))
            xy_density = natoms / xy_area
            
            # z方向密度  
            z_length = lattice.c
            z_density = natoms / z_length
            
            if xy_density > 0 and z_density > 0:
                anisotropy = abs(xy_density - z_density) / max(xy_density, z_density)
            else:
                anisotropy = 0
                
            return anisotropy
            
        except Exception:
            return 0
    
    def _convert_to_json_serializable(self, obj):
        """转换为JSON可序列化的格式"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        else:
            return obj
    
    def estimate_binding_energy(self, structure: Structure) -> float:
        """
        估算层间结合能
        基于简化的经验方法，不进行完整的DFT计算
        """
        try:
            # 使用经验公式估算范德华结合能
            # E_bind ≈ α * A / d^6 (其中A是接触面积，d是层间距离)
            
            interlayer_distances = self._calculate_interlayer_distances(structure)
            if not interlayer_distances:
                return 0.5  # 默认值
            
            avg_distance = np.mean(interlayer_distances)
            
            # 估算接触面积（xy平面面积）
            lattice = structure.lattice  
            contact_area = lattice.a * lattice.b * np.sin(np.radians(lattice.gamma))
            
            # 经验参数（基于石墨的值进行缩放）
            alpha = 0.1  # eV·Å⁴
            
            if avg_distance > 0:
                binding_energy = alpha * contact_area / (avg_distance ** 4)
                # 限制在合理范围内
                binding_energy = min(binding_energy, 1.0)  # 最大1.0 eV/Å²
            else:
                binding_energy = 0.5
                
            return binding_energy
            
        except Exception:
            return 0.5  # 默认中等结合能
    
    def process_single_material(self, cif_file: Path) -> Dict:
        """处理单个材料文件"""
        try:
            # 读取结构
            parser = CifParser(str(cif_file))
            structures = parser.parse_structures(primitive=True)
            if not structures:
                return {
                    'filename': cif_file.name,
                    'status': 'error',
                    'error': 'Failed to parse CIF file'
                }
            structure = structures[0]
            
            # 获取基本信息
            formula = structure.composition.reduced_formula
            elements = list(structure.composition.keys())
            num_elements = len(elements)
            
            # 检查元素数量约束
            if num_elements > self.max_elements:
                return {
                    'filename': cif_file.name,
                    'formula': formula,
                    'status': 'rejected',
                    'reason': f'Too many elements: {num_elements} > {self.max_elements}'
                }
            
            # 拓扑缩放分析
            tsa_result = self.topology_scaling_analysis(structure)
            
            # 层状结构分析
            layered_analysis = self.analyze_layered_structure(structure)
            
            # 估算结合能
            binding_energy = self.estimate_binding_energy(structure)
            
            # 综合评分
            score = self._calculate_2d_score(tsa_result, layered_analysis, binding_energy)
            
            # 判定结果（放宽条件）
            is_2d_candidate = (
                tsa_result['dimension'] == '2D' and
                binding_energy < self.max_binding_energy and
                score > self.min_2d_score
            )
            
            result = {
                'filename': cif_file.name,
                'formula': formula,
                'num_elements': int(num_elements),
                'elements': [str(el) for el in elements],
                'tsa_analysis': self._convert_to_json_serializable(tsa_result),
                'layered_analysis': self._convert_to_json_serializable(layered_analysis),
                'binding_energy_estimate': float(binding_energy),
                'score': float(score),
                'is_2d_candidate': bool(is_2d_candidate),
                'status': 'accepted' if is_2d_candidate else 'rejected'
            }
            
            # 更新统计
            if tsa_result['dimension'] == '2D':
                self.stats['dimensionality_2d'] += 1
            if binding_energy < self.max_binding_energy:
                self.stats['low_binding_energy'] += 1
            if layered_analysis['is_layered']:
                self.stats['layered_structure'] += 1
            if is_2d_candidate:
                self.stats['final_2d_candidates'] += 1
                
            return result
            
        except Exception as e:
            self.stats['processing_errors'] += 1
            return {
                'filename': cif_file.name,
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_2d_score(self, tsa_result: Dict, layered_analysis: Dict, binding_energy: float) -> float:
        """计算二维材料综合评分"""
        score = 0.0
        
        # TSA分析权重 (40%)
        if tsa_result['dimension'] == '2D':
            score += 0.4 * tsa_result['confidence']
        elif tsa_result['dimension'] == '1D':
            score += 0.1 * tsa_result['confidence']
        
        # 层状结构权重 (30%)
        if layered_analysis['is_layered']:
            score += 0.3
        
        # 层间距离权重 (20%)
        avg_distance = layered_analysis.get('avg_interlayer_distance', 0)
        if avg_distance > self.min_interlayer_distance:
            distance_score = min(1.0, (avg_distance - 2.0) / 3.0)  # 2-5Å范围内线性评分
            score += 0.2 * distance_score
        
        # 结合能权重 (10%)
        if binding_energy < self.max_binding_energy:
            binding_score = 1.0 - (binding_energy / self.max_binding_energy)
            score += 0.1 * binding_score
        
        return min(1.0, score)
    
    def run_screening(self) -> None:
        """运行筛选流程"""
        print("开始二维材料二次筛选...")
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        
        # 查找CIF文件
        cif_files = list(self.input_dir.glob("*.cif"))
        total_files = len(cif_files)
        
        if total_files == 0:
            print("未找到CIF文件！")
            return
        
        print(f"找到 {total_files} 个CIF文件")
        
        results = []
        
        # 处理每个文件
        for i, cif_file in enumerate(cif_files):
            if i % 100 == 0:
                print(f"处理进度: {i}/{total_files}")
            
            result = self.process_single_material(cif_file)
            results.append(result)
            self.stats['total_processed'] += 1
        
        # 保存结果
        self._save_results(results)
        self._print_statistics()
        
        print("筛选完成！")
    
    def _save_results(self, results: List[Dict]) -> None:
        """保存筛选结果"""
        # 保存完整结果
        results_file = self.output_dir / "2d_screening_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存二维材料候选
        candidates = [r for r in results if r.get('is_2d_candidate', False)]
        candidates_file = self.output_dir / "2d_material_candidates.json"
        with open(candidates_file, 'w', encoding='utf-8') as f:
            json.dump(candidates, f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式摘要
        df_data = []
        for result in results:
            if result['status'] != 'error':
                df_data.append({
                    'filename': result['filename'],
                    'formula': result['formula'],
                    'num_elements': result['num_elements'],
                    'tsa_dimension': result['tsa_analysis']['dimension'],
                    'tsa_confidence': result['tsa_analysis']['confidence'],
                    'is_layered': result['layered_analysis']['is_layered'],
                    'avg_interlayer_distance': result['layered_analysis'].get('avg_interlayer_distance', 0),
                    'binding_energy': result['binding_energy_estimate'],
                    'score': result['score'],
                    'is_2d_candidate': result['is_2d_candidate']
                })
        
        df = pd.DataFrame(df_data)
        df.to_csv(self.output_dir / "2d_screening_summary.csv", index=False)
        
        print(f"\n结果已保存:")
        print(f"- 完整结果: {results_file}")
        print(f"- 二维候选: {candidates_file}")
        print(f"- CSV摘要: {self.output_dir / '2d_screening_summary.csv'}")
    
    def _print_statistics(self) -> None:
        """打印统计信息"""
        print(f"\n=== 筛选统计 ===")
        print(f"总处理文件数: {self.stats['total_processed']}")
        print(f"TSA识别为2D: {self.stats['dimensionality_2d']}")
        print(f"低结合能材料: {self.stats['low_binding_energy']}")
        print(f"层状结构材料: {self.stats['layered_structure']}")
        print(f"最终2D候选: {self.stats['final_2d_candidates']}")
        print(f"处理错误: {self.stats['processing_errors']}")
        
        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['total_processed'] - self.stats['processing_errors']) / self.stats['total_processed']
            candidate_rate = self.stats['final_2d_candidates'] / self.stats['total_processed']
            print(f"处理成功率: {success_rate:.1%}")
            print(f"2D候选比例: {candidate_rate:.1%}")

def main():
    """主函数"""
    # 设置路径
    input_dir = "data/filtered"  # 已经过合成性筛选的材料
    output_dir = "data/2d_materials"
    
    # 创建筛选器
    screener = TwoDMaterialScreener(input_dir, output_dir)
    
    # 运行筛选
    screener.run_screening()

if __name__ == "__main__":
    main()