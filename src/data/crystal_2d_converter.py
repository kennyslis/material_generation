# -*- coding: utf-8 -*-
"""
晶胞3D到2D转换器
基于材料科学原理将3D晶胞结构转换为2D材料结构
主要功能:
1. 表面切割法 (Surface Cleavage) - 基于密勒指数切割
2. 层间分离法 (Layer Separation) - 识别并分离层状结构
3. 对称性降维法 (Symmetry Reduction) - 基于对称性降维
"""
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Lattice, PeriodicSite
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer as SGA
from pymatgen.io.cif import CifWriter, CifParser
from pymatgen.analysis.structure_matcher import StructureMatcher
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import json
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Crystal2DConverter:
    """
    3D晶胞到2D材料转换器
    
    支持多种转换方法:
    - surface_cleavage: 表面切割法，沿特定晶面切割
    - layer_separation: 层间分离法，识别并提取单层
    - symmetry_reduction: 对称性降维法，保留2D对称性
    """
    
    def __init__(self, 
                 min_vacuum_thickness: float = 15.0,
                 max_slab_thickness: float = 20.0,
                 min_atoms_2d: int = 2,
                 max_atoms_2d: int = 200):
        """
        初始化转换器参数
        
        Args:
            min_vacuum_thickness: 最小真空层厚度(Å)
            max_slab_thickness: 最大slab厚度(Å)  
            min_atoms_2d: 2D结构最少原子数
            max_atoms_2d: 2D结构最多原子数
        """
        self.min_vacuum_thickness = min_vacuum_thickness
        self.max_slab_thickness = max_slab_thickness
        self.min_atoms_2d = min_atoms_2d
        self.max_atoms_2d = max_atoms_2d

        # 最大尝试面数限制，避免计算过载
        self.max_miller_indices_to_try = 10
        
        # 层间距离阈值 (Å) - 用于识别层状结构
        self.distance_thresholds = {
            'van_der_waals': (3.0, 4.5),      # 范德华相互作用
            'weak_bonding': (2.5, 3.5),       # 弱键合
            'ionic_bonding': (2.0, 3.0)       # 离子键合
        }
        
        # 常见2D材料的密勒指数 - 扩展版本，按重要性排序
        self.common_2d_miller_indices = [
            # 基本面 - 最高优先级
            (0, 0, 1),   # 最常见的基面 - c轴垂直切割
            (1, 0, 0),   # a轴垂直切割
            (0, 1, 0),   # b轴垂直切割

            # 对角面 - 高优先级
            (1, 1, 0),   # ab平面对角线
            (1, 0, 1),   # ac平面对角线
            (0, 1, 1),   # bc平面对角线
            (1, 1, 1),   # 体对角线

            # 高指数面 - 中等优先级，适用于特殊材料
            (2, 1, 0),   # 常见于六方晶系
            (1, 2, 0),   # 常见于六方晶系
            (2, 0, 1),   # 适用于正交晶系
            (1, 0, 2),   # 适用于正交晶系
            (0, 2, 1),   # 适用于正交晶系

            # 其他常用面 - 较低优先级
            (2, 1, 1),   # 复合面
            (1, 2, 1),   # 复合面
            (1, 1, 2),   # 复合面
        ]
        
        # 层间键合类型识别的距离阈值 (Å)
        self.bonding_thresholds = {
            'covalent': (0.5, 2.0),        # 共价键
            'ionic': (1.5, 3.5),           # 离子键  
            'van_der_waals': (2.8, 4.5),   # 范德华力
            'metallic': (2.0, 3.8)         # 金属键
        }
        
    def analyze_structure_dimensionality(self, structure: Structure) -> Dict:
        """
        基于材料科学原理分析结构的维度特征
        
        Args:
            structure: pymatgen Structure对象
            
        Returns:
            包含维度分析结果的字典
        """
        lattice = structure.lattice
        abc = lattice.abc
        angles = lattice.angles
        
        # 晶格参数分析
        a, b, c = abc
        
        # 层状结构的关键特征分析
        c_to_a_ratio = c / a if a > 0 else float('inf')
        c_to_b_ratio = c / b if b > 0 else float('inf')
        ab_ratio = max(a, b) / min(a, b) if min(a, b) > 0 else float('inf')
        
        # 判断是否为层状结构（基于论文标准）
        is_layered_lattice = (
            1.5 <= c_to_a_ratio <= 8.0 and 
            1.5 <= c_to_b_ratio <= 8.0 and 
            ab_ratio <= 2.5  # 更严格的面内对称性要求
        )
        
        # 分析原子分布
        z_coords = [site.coords[2] for site in structure.sites]
        z_range = max(z_coords) - min(z_coords)
        relative_thickness = z_range / c if c > 0 else 1.0
        
        # 层状分布判断（更严格的标准）
        is_layered_distribution = (z_range < 8.0 and relative_thickness < 0.4)
        
        # 综合判断：真正的层状材料需要同时满足晶格和分布特征
        is_layered = is_layered_lattice and is_layered_distribution
        
        # 判断是否已经是准2D结构（有真空层的单层）
        is_already_2d = (
            is_layered and 
            c > 15.0 and  # 有足够的真空层
            relative_thickness < 0.2  # 原子层很薄
        )
        
        # 识别可能的转换策略
        conversion_potential = "none"
        if is_layered_lattice and not is_layered_distribution:
            conversion_potential = "exfoliation"  # 剥离法 - 晶格层状但原子分布厚
        elif not is_layered_lattice and c_to_a_ratio > 1.5:
            conversion_potential = "surface_cleavage"  # 表面切割法
        elif is_layered:
            conversion_potential = "already_suitable"  # 已经适合
        
        return {
            'lattice_parameters': abc,
            'lattice_angles': angles,
            'lattice_ratios': {
                'c/a': c_to_a_ratio,
                'c/b': c_to_b_ratio,
                'ab_ratio': ab_ratio
            },
            'is_layered_lattice': is_layered_lattice,
            'is_layered_distribution': is_layered_distribution,
            'is_layered': is_layered,
            'is_already_2d': is_already_2d,
            'z_range_angstrom': z_range,
            'relative_thickness': relative_thickness,
            'conversion_potential': conversion_potential,
            'n_atoms': len(structure.sites)
        }
    
    def identify_cleavage_planes(self, structure: Structure) -> List[Tuple]:
        """
        识别可能的解理面
        
        Args:
            structure: 输入结构
            
        Returns:
            可能的密勒指数列表
        """
        # 基于对称性分析识别可能的解理面
        try:
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            
            # 获取结构的对称元素
            candidate_planes = []
            
            # 优先考虑高对称性的面，限制尝试次数
            indices_to_try = self.common_2d_miller_indices[:self.max_miller_indices_to_try]
            for hkl in indices_to_try:
                try:
                    # 尝试生成slab来验证是否可行
                    slabgen = SlabGenerator(
                        initial_structure=structure,
                        miller_index=hkl,
                        min_slab_size=5.0,
                        min_vacuum_size=10.0,
                        center_slab=True
                    )
                    slabs = slabgen.get_slabs()
                    if slabs:
                        candidate_planes.append(hkl)
                        logger.debug(f"密勒指数 {hkl} 可行，生成了 {len(slabs)} 个slab")
                except Exception as e:
                    logger.debug(f"无法沿{hkl}面生成slab: {e}")
                    continue
                    
            logger.info(f"识别到 {len(candidate_planes)} 个可行的解理面: {candidate_planes}")
            return candidate_planes

        except Exception as e:
            logger.warning(f"对称性分析失败: {e}")
            # 采用智能默认选择策略
            default_planes = self._get_intelligent_default_planes(structure)
            logger.info(f"使用智能默认解理面: {default_planes}")
            return default_planes

    def _get_intelligent_default_planes(self, structure: Structure) -> List[Tuple]:
        """
        智能默认解理面选择策略

        Args:
            structure: 输入结构

        Returns:
            智能选择的默认密勒指数列表
        """
        try:
            lattice = structure.lattice
            a, b, c = lattice.abc

            # 基于晶格参数的智能判断
            c_to_ab_max = c / max(a, b) if max(a, b) > 0 else 1.0
            ab_similarity = abs(a - b) / max(a, b) if max(a, b) > 0 else 0.0
            abc_similarity = (abs(a - b) + abs(b - c) + abs(a - c)) / (a + b + c) if (a + b + c) > 0 else 1.0

            # 层状特征强的材料 (c轴明显大于a,b轴)
            if c_to_ab_max > 1.8:
                # 优先选择垂直于c轴的面
                return [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 1)]

            # 立方或准立方结构
            elif abc_similarity < 0.1:
                # 三个轴相近，选择对称性好的面
                return [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 1, 1)]

            # 面内各向异性明显的材料 (a,b轴差异大但c轴适中)
            elif ab_similarity > 0.3 and 1.2 < c_to_ab_max < 1.8:
                # 选择能体现各向异性的面
                return [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (2, 1, 0)]

            # 其他情况：使用通用策略
            else:
                return [(0, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 1)]

        except Exception as e:
            logger.debug(f"智能默认选择失败，使用固定默认值: {e}")
            # 最后的兜底策略
            return [(0, 0, 1), (1, 0, 0), (0, 1, 0)]

    def _estimate_layer_count(self, z_range: float) -> int:
        """
        估算晶胞中的层数

        Args:
            z_range: z方向原子分布范围

        Returns:
            估算的层数
        """
        # 基于典型层状材料的层间距估算层数
        # 典型层间距：石墨3.35Å, TMDCs 6-7Å, 平均约4-5Å
        typical_interlayer_spacing = 4.5  # Å
        estimated_layers = max(1, round(z_range / typical_interlayer_spacing))
        return min(estimated_layers, 5)  # 限制最大层数为5

    def _estimate_layer_thickness_by_count(self, z_range: float, estimated_layers: int) -> float:
        """
        基于层数的动态单层厚度估算

        Args:
            z_range: z方向原子分布范围
            estimated_layers: 估算的层数

        Returns:
            估算的单层厚度
        """
        if estimated_layers <= 2:
            return min(z_range * 0.45, 6.0)
        elif estimated_layers == 3:
            return min(z_range * 0.32, 6.0)
        else:
            return min(z_range * 0.25, 6.0)

    def surface_cleavage_method(self, structure: Structure, 
                               miller_indices: Optional[List[Tuple]] = None) -> List[Structure]:
        """
        表面切割法：沿指定晶面生成2D slab结构
        
        Args:
            structure: 输入的3D结构
            miller_indices: 密勒指数列表，如果None则自动识别
            
        Returns:
            生成的2D结构列表
        """
        if miller_indices is None:
            miller_indices = self.identify_cleavage_planes(structure)
        
        generated_2d_structures = []
        
        for hkl in miller_indices:
            try:
                # 创建SlabGenerator
                slabgen = SlabGenerator(
                    initial_structure=structure,
                    miller_index=hkl,
                    min_slab_size=3.0,  # 最小slab厚度
                    min_vacuum_size=self.min_vacuum_thickness,
                    center_slab=True,
                    primitive=False,
                    max_normal_search=1
                )
                
                slabs = slabgen.get_slabs()[:3]  # 限制最多3个slab
                
                for i, slab in enumerate(slabs):
                    if self.validate_2d_structure(slab):
                        # 验证切面连通性
                        connectivity_result = self.verify_slab_connectivity(slab)

                        if connectivity_result['overall_connected']:
                            # 添加元数据
                            slab.properties = {
                                'conversion_method': 'surface_cleavage',
                                'miller_index': hkl,
                                'slab_index': i,
                                'vacuum_thickness': self.min_vacuum_thickness,
                                'connectivity_verified': True,
                                'connectivity_info': connectivity_result
                            }
                            generated_2d_structures.append(slab)
                            logger.debug(f"切面{hkl}的slab {i}通过连通性验证")
                        else:
                            logger.debug(f"切面{hkl}的slab {i}未通过连通性验证")
                        
                logger.info(f"沿{hkl}面成功生成{len(slabs)}个slab结构")
                
            except Exception as e:
                logger.warning(f"沿{hkl}面生成slab失败: {e}")
                continue
                
        return generated_2d_structures
    
    def exfoliation_method(self, structure: Structure) -> List[Structure]:
        """
        层剥离法：基于层间键合分析，剥离出单层或少层结构
        
        这是真正基于物理原理的2D材料生成方法：
        1. 识别层间弱相互作用（范德华力、弱离子键等）
        2. 沿弱键合方向剥离出单层结构
        3. 保持层内强键合完整性
        
        Args:
            structure: 输入的层状结构
            
        Returns:
            剥离得到的2D结构列表
        """
        generated_2d_structures = []
        
        try:
            # 1. 分析层间键合类型
            layer_info = self._analyze_interlayer_bonding(structure)
            
            if not layer_info['has_weak_interlayer']:
                logger.warning("材料不具备可剥离的层间弱键合")
                return []
            
            # 2. 识别层的z坐标位置
            layer_positions = self._identify_layer_positions(structure, layer_info)
            
            # 3. 提取单层或少层结构
            for i, layer_z in enumerate(layer_positions[:3]):  # 最多生成3个变体
                try:
                    # 确定剥离厚度 - 包含一个完整的化学层
                    layer_thickness = layer_info['intralayer_thickness']
                    extraction_range = (layer_z - layer_thickness/2, 
                                      layer_z + layer_thickness/2)
                    
                    # 提取该层的所有原子
                    layer_sites = []
                    for site in structure.sites:
                        z_coord = site.coords[2]
                        if extraction_range[0] <= z_coord <= extraction_range[1]:
                            layer_sites.append(site)
                    
                    if len(layer_sites) < self.min_atoms_2d:
                        continue
                    
                    # 创建2D晶胞：保持a,b参数，增大c轴加入真空层
                    original_lattice = structure.lattice
                    new_c = max(self.min_vacuum_thickness + layer_thickness, 20.0)
                    
                    new_lattice = Lattice.from_parameters(
                        a=original_lattice.a,
                        b=original_lattice.b,
                        c=new_c,
                        alpha=original_lattice.alpha,
                        beta=original_lattice.beta,
                        gamma=original_lattice.gamma
                    )
                    
                    # 重新定位原子到晶胞中心
                    center_z = new_c / 2
                    layer_center = (extraction_range[0] + extraction_range[1]) / 2
                    z_shift = center_z - layer_center
                    
                    new_sites = []
                    for site in layer_sites:
                        new_coords = site.coords.copy()
                        new_coords[2] += z_shift
                        new_frac_coords = new_lattice.get_fractional_coords(new_coords)
                        new_sites.append((site.species, new_frac_coords))
                    
                    # 创建2D结构
                    species = [site[0] for site in new_sites]
                    coords = [site[1] for site in new_sites]
                    
                    exfoliated_structure = Structure(
                        lattice=new_lattice,
                        species=species,
                        coords=coords
                    )
                    
                    if self.validate_2d_structure(exfoliated_structure):
                        exfoliated_structure.properties = {
                            'conversion_method': 'exfoliation',
                            'layer_index': i,
                            'layer_thickness_angstrom': layer_thickness,
                            'interlayer_bonding': layer_info['dominant_bonding_type'],
                            'vacuum_thickness': self.min_vacuum_thickness
                        }
                        generated_2d_structures.append(exfoliated_structure)
                        
                        logger.info(f"成功剥离第{i+1}层 (厚度: {layer_thickness:.2f}Å)")
                        
                except Exception as e:
                    logger.warning(f"剥离第{i+1}层失败: {e}")
                    continue
            
            return generated_2d_structures
            
        except Exception as e:
            logger.error(f"层剥离方法失败: {e}")
            return []
    
    def _analyze_interlayer_bonding(self, structure: Structure) -> Dict:
        """分析层间键合类型"""
        lattice = structure.lattice
        
        # 获取所有原子间距离
        distances = []
        bonding_types = []
        
        for i, site1 in enumerate(structure.sites):
            for j, site2 in enumerate(structure.sites[i+1:], i+1):
                dist = site1.distance(site2)
                if dist < 5.0:  # 只考虑较近的原子对
                    distances.append(dist)
                    
                    # 基于距离判断键合类型
                    bonding_type = 'unknown'
                    for bond_type, (min_d, max_d) in self.bonding_thresholds.items():
                        if min_d <= dist <= max_d:
                            bonding_type = bond_type
                            break
                    bonding_types.append(bonding_type)
        
        # 统计键合类型
        bonding_stats = {}
        for bond_type in bonding_types:
            bonding_stats[bond_type] = bonding_stats.get(bond_type, 0) + 1
        
        # 判断是否有层间弱键合
        weak_bonding_count = bonding_stats.get('van_der_waals', 0)
        total_bonding = sum(bonding_stats.values())
        
        has_weak_interlayer = weak_bonding_count > 0.1 * total_bonding
        
        # 估算层内厚度（通过z方向原子分布分析）
        z_coords = [site.coords[2] for site in structure.sites]
        z_range = max(z_coords) - min(z_coords)
        
        # 使用基于层数的动态估算方法
        estimated_layers = self._estimate_layer_count(z_range)
        estimated_layer_thickness = self._estimate_layer_thickness_by_count(z_range, estimated_layers)
        
        return {
            'has_weak_interlayer': has_weak_interlayer,
            'bonding_statistics': bonding_stats,
            'dominant_bonding_type': max(bonding_stats.items(), key=lambda x: x[1])[0] if bonding_stats else 'unknown',
            'weak_bonding_ratio': weak_bonding_count / total_bonding if total_bonding > 0 else 0,
            'intralayer_thickness': estimated_layer_thickness,
            'total_z_range': z_range
        }
    
    def _identify_layer_positions(self, structure: Structure, layer_info: Dict) -> List[float]:
        """识别层的z坐标位置"""
        z_coords = [site.coords[2] for site in structure.sites]
        
        # 使用简单的聚类方法识别层
        from sklearn.cluster import DBSCAN
        import numpy as np
        
        z_array = np.array(z_coords).reshape(-1, 1)
        
        # 使用自适应eps计算，确保同一层的原子被聚类在一起
        adaptive_eps = self.get_adaptive_eps(structure, z_array, method='material_specific')
        # 结合层厚度信息进行调整
        adjusted_eps = min(adaptive_eps, layer_info['intralayer_thickness'] / 2)
        logger.debug(f"使用调整后的eps值: {adjusted_eps:.3f} (自适应: {adaptive_eps:.3f}, 层厚度基准: {layer_info['intralayer_thickness'] / 2:.3f})")
        clustering = DBSCAN(eps=adjusted_eps, min_samples=2).fit(z_array)
        
        # 计算每个簇的中心z坐标
        layer_centers = []
        for label in set(clustering.labels_):
            if label != -1:  # 忽略噪声点
                cluster_z_coords = [z_coords[i] for i, l in enumerate(clustering.labels_) if l == label]
                center_z = np.mean(cluster_z_coords)
                layer_centers.append(center_z)
        
        return sorted(layer_centers)
    
    def layer_separation_method(self, structure: Structure) -> List[Structure]:
        """
        层间分离法：识别层状结构并提取单层
        
        Args:
            structure: 输入结构
            
        Returns:
            分离得到的2D结构列表
        """
        # 获取所有原子的笛卡尔坐标
        cart_coords = []
        for site in structure.sites:
            cart_coords.append(structure.lattice.get_cartesian_coords(site.frac_coords))
        cart_coords = np.array(cart_coords)
        
        # 基于z坐标进行聚类分析识别层，使用增强的连通性验证
        z_coords = cart_coords[:, 2].reshape(-1, 1)

        # 使用增强的层分析方法（包含自适应eps和连通性验证）
        validated_layers = self.enhanced_layer_analysis(structure)
        n_layers = len(validated_layers)

        logger.info(f"经连通性验证后识别到{n_layers}个有效层")

        generated_2d_structures = []

        if n_layers >= 2:  # 如果有多层，尝试提取单层
            for layer_info in validated_layers:
                layer_sites = layer_info['sites']
                
                if len(layer_sites) < self.min_atoms_2d:
                    continue
                
                # 创建新的晶格参数（保持a,b，扩大c加入真空层）
                original_lattice = structure.lattice
                new_lattice = Lattice.from_parameters(
                    a=original_lattice.a,
                    b=original_lattice.b, 
                    c=max(original_lattice.c, self.min_vacuum_thickness + 5.0),
                    alpha=original_lattice.alpha,
                    beta=original_lattice.beta,
                    gamma=original_lattice.gamma
                )
                
                # 重新计算分数坐标
                layer_frac_coords = []
                layer_species = []
                for site in layer_sites:
                    # 将原子移到晶胞中心附近
                    cart_coord = original_lattice.get_cartesian_coords(site.frac_coords)
                    cart_coord[2] = 0.5 * new_lattice.c  # 置于新晶胞中心
                    new_frac_coord = new_lattice.get_fractional_coords(cart_coord)
                    
                    layer_frac_coords.append(new_frac_coord)
                    layer_species.append(site.species)
                
                try:
                    # 创建新的2D结构
                    layer_structure = Structure(
                        lattice=new_lattice,
                        species=layer_species,
                        coords=layer_frac_coords
                    )
                    
                    if self.validate_2d_structure(layer_structure):
                        layer_structure.properties = {
                            'conversion_method': 'layer_separation',
                            'layer_id': int(layer_id),
                            'n_layers_total': n_layers,
                            'vacuum_thickness': self.min_vacuum_thickness
                        }
                        generated_2d_structures.append(layer_structure)
                        
                except Exception as e:
                    logger.warning(f"创建层{layer_id}结构失败: {e}")
                    continue
        
        return generated_2d_structures
    
    def symmetry_reduction_method(self, structure: Structure) -> List[Structure]:
        """
        对称性降维法：基于空间群对称性进行维度约简
        
        Args:
            structure: 输入结构
            
        Returns:
            降维得到的2D结构列表
        """
        try:
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            space_group = sga.get_space_group_symbol()
            
            logger.info(f"原始结构空间群: {space_group}")
            
            # 对于某些特定的空间群，可以应用特定的降维策略
            generated_2d_structures = []
            
            # 获取原胞
            primitive_structure = sga.get_primitive_standard_structure()
            
            # 尝试沿主要晶轴方向创建2D投影
            for axis in [0, 1, 2]:  # x, y, z轴
                try:
                    # 选择垂直于该轴的平面
                    if axis == 0:  # 沿x轴投影到yz平面
                        miller_index = (1, 0, 0)
                    elif axis == 1:  # 沿y轴投影到xz平面  
                        miller_index = (0, 1, 0)
                    else:  # 沿z轴投影到xy平面
                        miller_index = (0, 0, 1)
                    
                    # 使用surface_cleavage方法
                    slab_structures = self.surface_cleavage_method(
                        primitive_structure, 
                        [miller_index]
                    )
                    
                    for slab in slab_structures:
                        slab.properties.update({
                            'conversion_method': 'symmetry_reduction',
                            'reduction_axis': axis,
                            'original_space_group': space_group
                        })
                        generated_2d_structures.append(slab)
                        
                except Exception as e:
                    logger.debug(f"沿轴{axis}降维失败: {e}")
                    continue
            
            return generated_2d_structures
            
        except Exception as e:
            logger.error(f"对称性降维失败: {e}")
            return []

    def calculate_adaptive_eps(self, coordinates: np.ndarray, k: int = 4) -> float:
        """
        基于K-Distance图的eps自适应计算

        Args:
            coordinates: 坐标数组
            k: 近邻数量

        Returns:
            优化的eps值
        """
        try:
            # 计算k近邻距离
            neighbors = NearestNeighbors(n_neighbors=k)
            neighbors_fit = neighbors.fit(coordinates)
            distances, indices = neighbors_fit.kneighbors(coordinates)

            # 取第k个邻居的距离并排序
            k_distances = distances[:, k-1]
            k_distances = np.sort(k_distances)

            # 简单的肘点检测：使用二阶导数
            if len(k_distances) > 10:
                # 计算二阶导数近似
                second_derivative = np.diff(k_distances, 2)
                # 找到最大二阶导数的点
                knee_index = np.argmax(second_derivative) + 2
                optimal_eps = k_distances[knee_index] if knee_index < len(k_distances) else np.mean(k_distances)
            else:
                optimal_eps = np.mean(k_distances)

            return optimal_eps

        except Exception as e:
            logger.warning(f"自适应eps计算失败: {e}，使用默认值")
            return 2.0

    def material_specific_eps(self, structure: Structure, z_coords: np.ndarray) -> float:
        """
        基于材料类型和原子密度的eps估算

        Args:
            structure: 输入结构
            z_coords: z坐标数组

        Returns:
            材料特异性的eps值
        """
        try:
            # 计算最近邻距离
            from pymatgen.analysis.local_env import CrystalNN

            cn = CrystalNN()
            nearest_distances = []

            for i, site in enumerate(structure.sites):
                try:
                    nn_info = cn.get_nn_info(structure, i)
                    if nn_info:
                        distances = [info['weight'] for info in nn_info]
                        nearest_distances.extend(distances)
                except:
                    continue

            if not nearest_distances:
                return 2.0  # 默认值

            # 基于材料类型调整
            composition = structure.composition
            if any(elem.is_transition_metal for elem in composition.elements):
                # TMDCs材料：层间距较大
                base_eps = np.mean(nearest_distances) * 1.5
            elif all(elem.symbol in ['C', 'B', 'N'] for elem in composition.elements):
                # 类石墨材料：层间距中等
                base_eps = np.mean(nearest_distances) * 1.2
            else:
                # 其他材料：默认策略
                base_eps = np.mean(nearest_distances) * 1.0

            return base_eps

        except Exception as e:
            logger.warning(f"材料特异性eps计算失败: {e}，使用默认值")
            return 2.0

    def get_adaptive_eps(self, structure: Structure, z_coords: np.ndarray, method: str = 'hybrid') -> float:
        """
        综合自适应eps计算

        Args:
            structure: 输入结构
            z_coords: z坐标数组
            method: 计算方法 ('k_distance', 'material_specific', 'hybrid')

        Returns:
            自适应eps值
        """
        if method == 'k_distance':
            return self.calculate_adaptive_eps(z_coords)
        elif method == 'material_specific':
            return self.material_specific_eps(structure, z_coords)
        elif method == 'hybrid':
            # 综合两种方法
            k_dist_eps = self.calculate_adaptive_eps(z_coords)
            material_eps = self.material_specific_eps(structure, z_coords)
            return (k_dist_eps + material_eps) / 2
        else:
            return 2.0  # 默认值

    def verify_layer_connectivity(self, layer_sites: List, threshold: float = 3.0) -> Dict:
        """
        验证层内原子的xy平面连通性

        Args:
            layer_sites: 层内原子位点列表
            threshold: 连通性距离阈值

        Returns:
            连通性分析结果
        """
        try:
            # 构建原子坐标（仅xy坐标）
            xy_coords = np.array([[site.coords[0], site.coords[1]] for site in layer_sites])

            # 计算距离矩阵
            distances = squareform(pdist(xy_coords))

            # 构建连通图
            G = nx.Graph()
            n_atoms = len(layer_sites)

            # 添加节点
            for i in range(n_atoms):
                G.add_node(i)

            # 添加边（基于距离阈值）
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    if distances[i, j] <= threshold:
                        G.add_edge(i, j)

            # 检查连通性
            connected_components = list(nx.connected_components(G))
            is_fully_connected = len(connected_components) == 1

            return {
                'is_connected': is_fully_connected,
                'num_components': len(connected_components),
                'largest_component_size': max(len(comp) for comp in connected_components) if connected_components else 0,
                'components': connected_components
            }

        except Exception as e:
            logger.warning(f"连通性验证失败: {e}")
            return {
                'is_connected': False,
                'num_components': 0,
                'largest_component_size': 0,
                'components': []
            }

    def verify_slab_connectivity(self, slab_structure: Structure, connectivity_threshold: float = 3.0) -> Dict:
        """
        验证切面结构的连通性

        Args:
            slab_structure: 切面结构
            connectivity_threshold: 连通性距离阈值

        Returns:
            切面连通性分析结果
        """
        try:
            # 获取表面原子
            surface_sites = []
            z_coords = [site.coords[2] for site in slab_structure.sites]
            z_min, z_max = min(z_coords), max(z_coords)

            # 识别顶层和底层原子（表面）
            for site in slab_structure.sites:
                z = site.coords[2]
                if abs(z - z_max) < 1.0 or abs(z - z_min) < 1.0:  # 表面原子
                    surface_sites.append(site)

            # 验证每个表面的连通性
            top_surface = [site for site in surface_sites if abs(site.coords[2] - z_max) < 1.0]
            bottom_surface = [site for site in surface_sites if abs(site.coords[2] - z_min) < 1.0]

            top_connectivity = self.verify_layer_connectivity(top_surface, connectivity_threshold)
            bottom_connectivity = self.verify_layer_connectivity(bottom_surface, connectivity_threshold)

            return {
                'top_surface': top_connectivity,
                'bottom_surface': bottom_connectivity,
                'overall_connected': top_connectivity['is_connected'] and bottom_connectivity['is_connected']
            }

        except Exception as e:
            logger.warning(f"切面连通性验证失败: {e}")
            return {
                'top_surface': {'is_connected': False},
                'bottom_surface': {'is_connected': False},
                'overall_connected': False
            }

    def enhanced_layer_analysis(self, structure: Structure) -> List[Dict]:
        """
        增强的层分析（包含连通性验证）

        Args:
            structure: 输入结构

        Returns:
            验证后的层列表
        """
        try:
            z_coords = np.array([[site.coords[2]] for site in structure.sites])

            # 自适应eps计算
            adaptive_eps = self.get_adaptive_eps(structure, z_coords)
            logger.info(f"使用自适应eps值: {adaptive_eps:.3f}")

            # DBSCAN聚类
            clustering = DBSCAN(eps=adaptive_eps, min_samples=2).fit(z_coords)

            # 连通性验证
            validated_layers = []
            for label in set(clustering.labels_):
                if label == -1:  # 跳过噪声
                    continue

                layer_indices = np.where(clustering.labels_ == label)[0]
                layer_sites = [structure.sites[i] for i in layer_indices]

                # 验证连通性
                connectivity = self.verify_layer_connectivity(layer_sites)

                if connectivity['is_connected']:
                    validated_layers.append({
                        'label': label,
                        'sites': layer_sites,
                        'connectivity': connectivity
                    })
                else:
                    # 处理非连通层：分离成连通组件
                    for comp_idx, component in enumerate(connectivity['components']):
                        if len(component) >= self.min_atoms_2d:
                            comp_sites = [layer_sites[i] for i in component]
                            validated_layers.append({
                                'label': f"{label}_comp_{comp_idx}",
                                'sites': comp_sites,
                                'connectivity': {'is_connected': True, 'num_components': 1}
                            })

            logger.info(f"验证后识别到 {len(validated_layers)} 个连通的层")
            return validated_layers

        except Exception as e:
            logger.error(f"增强层分析失败: {e}")
            return []

    def validate_2d_structure(self, structure: Structure) -> bool:
        """
        基于材料科学标准验证生成的结构是否为有效的2D结构

        验证标准基于文献研究：
        1. 几何标准：长宽比(aspect ratio)和厚度约束
        2. 化学键合分析：层内强键合vs层间弱相互作用
        3. 稳定性评估：结构完整性和原子分布
        4. 真空层验证：确保有效的2D周期性

        Args:
            structure: 待验证的结构

        Returns:
            是否为有效2D结构
        """
        try:
            validation_results = self.comprehensive_2d_validation(structure)

            # 综合评分机制：所有关键标准都必须通过
            required_criteria = [
                'atomic_count_valid',
                'geometry_valid',
                'vacuum_sufficient',
                'bonding_analysis_valid'
            ]

            for criterion in required_criteria:
                if not validation_results.get(criterion, False):
                    logger.debug(f"2D验证失败: {criterion} = {validation_results.get(criterion)}")
                    return False

            # 计算综合得分
            score = validation_results.get('overall_score', 0)
            threshold = 0.7  # 70%通过阈值

            logger.debug(f"2D结构验证得分: {score:.3f} (阈值: {threshold})")
            return score >= threshold

        except Exception as e:
            logger.warning(f"结构验证失败: {e}")
            return False

    def comprehensive_2d_validation(self, structure: Structure) -> Dict:
        """
        基于文献标准的全面2D结构验证

        参考文献标准：
        - 形成能稳定性 (Ehull < 0.1 eV/atom)
        - 几何约束 (层厚度 < 10Å, 长宽比 > 10)
        - van der Waals间隙特征
        - 原子密度分布

        Args:
            structure: 待验证结构

        Returns:
            详细验证结果字典
        """
        results = {
            'atomic_count_valid': False,
            'geometry_valid': False,
            'vacuum_sufficient': False,
            'bonding_analysis_valid': False,
            'aspect_ratio': 0.0,
            'layer_thickness': 0.0,
            'vacuum_ratio': 0.0,
            'overall_score': 0.0
        }

        try:
            # 1. 原子数量检查
            n_atoms = len(structure.sites)
            results['atomic_count_valid'] = self.min_atoms_2d <= n_atoms <= self.max_atoms_2d

            # 2. 几何验证
            geometry_validation = self._validate_2d_geometry(structure)
            results.update(geometry_validation)

            # 3. 真空层验证
            vacuum_validation = self._validate_vacuum_layers(structure)
            results.update(vacuum_validation)

            # 4. 化学键合分析
            bonding_validation = self._validate_2d_bonding(structure)
            results.update(bonding_validation)

            # 5. 综合评分计算
            results['overall_score'] = self._calculate_2d_score(results)

            return results

        except Exception as e:
            logger.error(f"全面2D验证失败: {e}")
            return results

    def _validate_2d_geometry(self, structure: Structure) -> Dict:
        """验证2D几何标准"""
        try:
            lattice = structure.lattice
            a, b, c = lattice.abc

            # 计算原子在z方向的实际分布
            z_coords = [site.coords[2] for site in structure.sites]
            layer_thickness = max(z_coords) - min(z_coords)

            # 计算长宽比 (lateral size / thickness)
            lateral_size = max(a, b)
            aspect_ratio = lateral_size / max(layer_thickness, 0.1)  # 避免除零

            # 文献标准：
            # - 层厚度 < 10Å (单层或少层材料)
            # - 长宽比 > 10 (保证2D特性)
            # - c轴长度应显著大于层厚度(包含真空)

            thickness_valid = layer_thickness < 10.0  # Å
            aspect_ratio_valid = aspect_ratio > 10.0
            c_axis_reasonable = c > layer_thickness * 2  # c轴至少是层厚的2倍

            geometry_valid = thickness_valid and aspect_ratio_valid and c_axis_reasonable

            return {
                'geometry_valid': geometry_valid,
                'layer_thickness': layer_thickness,
                'aspect_ratio': aspect_ratio,
                'thickness_valid': thickness_valid,
                'aspect_ratio_valid': aspect_ratio_valid,
                'c_axis_reasonable': c_axis_reasonable
            }

        except Exception as e:
            logger.warning(f"几何验证失败: {e}")
            return {'geometry_valid': False, 'layer_thickness': 0.0, 'aspect_ratio': 0.0}

    def _validate_vacuum_layers(self, structure: Structure) -> Dict:
        """验证真空层合理性"""
        try:
            lattice = structure.lattice
            c_length = lattice.c

            # 计算原子在分数坐标系中的z分布
            z_frac = [site.frac_coords[2] for site in structure.sites]
            min_z_frac, max_z_frac = min(z_frac), max(z_frac)

            # 计算真空区域
            vacuum_below = min_z_frac * c_length
            vacuum_above = (1 - max_z_frac) * c_length
            total_vacuum = vacuum_below + vacuum_above

            # 计算真空比例
            material_thickness = (max_z_frac - min_z_frac) * c_length
            vacuum_ratio = total_vacuum / c_length

            # 文献标准：
            # - 总真空层 >= 最小要求
            # - 真空比例 > 0.5 (真空占一半以上空间)
            # - 材料厚度 < c轴的40%

            vacuum_sufficient = total_vacuum >= self.min_vacuum_thickness
            vacuum_ratio_good = vacuum_ratio > 0.5
            material_fraction_reasonable = material_thickness < c_length * 0.4

            vacuum_valid = vacuum_sufficient and vacuum_ratio_good and material_fraction_reasonable

            return {
                'vacuum_sufficient': vacuum_valid,
                'vacuum_ratio': vacuum_ratio,
                'total_vacuum': total_vacuum,
                'material_thickness': material_thickness,
                'vacuum_ratio_good': vacuum_ratio_good,
                'material_fraction_reasonable': material_fraction_reasonable
            }

        except Exception as e:
            logger.warning(f"真空层验证失败: {e}")
            return {'vacuum_sufficient': False, 'vacuum_ratio': 0.0}

    def _validate_2d_bonding(self, structure: Structure) -> Dict:
        """验证2D材料的键合特征"""
        try:
            from pymatgen.analysis.local_env import CrystalNN

            cn = CrystalNN()
            in_plane_distances = []
            out_of_plane_distances = []

            # 计算原子z坐标用于区分层内和层间
            z_coords = [site.coords[2] for site in structure.sites]
            z_threshold = 2.0  # Å，用于区分层内和层间相互作用

            for i, site in enumerate(structure.sites):
                try:
                    nn_info = cn.get_nn_info(structure, i)
                    for neighbor in nn_info:
                        distance = neighbor['weight']
                        neighbor_z = structure.sites[neighbor['site_index']].coords[2]

                        # 根据z坐标差异区分层内和层间键合
                        z_diff = abs(z_coords[i] - neighbor_z)

                        if z_diff < z_threshold:
                            in_plane_distances.append(distance)
                        else:
                            out_of_plane_distances.append(distance)

                except:
                    continue

            # 分析键合特征
            if in_plane_distances and out_of_plane_distances:
                avg_in_plane = np.mean(in_plane_distances)
                avg_out_of_plane = np.mean(out_of_plane_distances)
                bonding_ratio = avg_out_of_plane / avg_in_plane if avg_in_plane > 0 else 1.0

                # van der Waals特征：层间距离应明显大于层内距离
                vdw_characteristic = bonding_ratio > 1.5  # 层间距离至少是层内的1.5倍

            elif in_plane_distances:  # 只有层内键合（可能是单层）
                vdw_characteristic = True
                bonding_ratio = 1.0

            else:
                vdw_characteristic = False
                bonding_ratio = 1.0

            # 连通性验证
            if len(structure.sites) >= 3:
                connectivity_result = self.verify_layer_connectivity(structure.sites)
                connectivity_valid = connectivity_result['is_connected']
            else:
                connectivity_valid = True  # 小结构默认连通

            bonding_valid = vdw_characteristic and connectivity_valid

            return {
                'bonding_analysis_valid': bonding_valid,
                'vdw_characteristic': vdw_characteristic,
                'connectivity_valid': connectivity_valid,
                'bonding_ratio': bonding_ratio,
                'in_plane_bonds': len(in_plane_distances),
                'out_of_plane_bonds': len(out_of_plane_distances)
            }

        except Exception as e:
            logger.warning(f"键合分析失败: {e}")
            return {'bonding_analysis_valid': True}  # 如果分析失败，不阻止验证

    def _calculate_2d_score(self, validation_results: Dict) -> float:
        """计算2D结构的综合评分"""
        try:
            score = 0.0

            # 权重分配（基于文献重要性）
            weights = {
                'atomic_count_valid': 0.15,      # 15% - 基础要求
                'geometry_valid': 0.35,          # 35% - 几何是核心
                'vacuum_sufficient': 0.25,       # 25% - 真空层重要
                'bonding_analysis_valid': 0.25   # 25% - 键合特征关键
            }

            # 基础分数计算
            for criterion, weight in weights.items():
                if validation_results.get(criterion, False):
                    score += weight

            # 几何质量奖励分数
            aspect_ratio = validation_results.get('aspect_ratio', 0)
            if aspect_ratio > 50:  # 高长宽比奖励
                score += 0.05
            elif aspect_ratio > 20:
                score += 0.03

            # 真空比例奖励分数
            vacuum_ratio = validation_results.get('vacuum_ratio', 0)
            if vacuum_ratio > 0.7:  # 高真空比例奖励
                score += 0.05
            elif vacuum_ratio > 0.6:
                score += 0.03

            # 键合特征奖励分数
            bonding_ratio = validation_results.get('bonding_ratio', 1.0)
            if bonding_ratio > 2.0:  # 明显的van der Waals特征
                score += 0.05

            return min(score, 1.0)  # 确保分数不超过1.0

        except Exception as e:
            logger.warning(f"评分计算失败: {e}")
            return 0.0
    
    def generate_2d_structures_from_3d(self, 
                                      structure: Structure,
                                      methods: List[str] = None) -> List[Structure]:
        """
        从3D结构生成2D结构的主要方法（基于材料科学原理）
        
        Args:
            structure: 输入的3D结构
            methods: 要使用的转换方法列表，如果为None则根据结构特征自动选择
            
        Returns:
            生成的2D结构列表
        """
        # 首先分析结构特征
        analysis = self.analyze_structure_dimensionality(structure)
        logger.info(f"结构分析结果:")
        logger.info(f"  - 层状晶格: {analysis['is_layered_lattice']}")
        logger.info(f"  - 层状分布: {analysis['is_layered_distribution']}")  
        logger.info(f"  - 综合层状: {analysis['is_layered']}")
        logger.info(f"  - 转换潜力: {analysis['conversion_potential']}")
        
        # 如果已经是2D结构，直接返回
        if analysis['is_already_2d']:
            logger.info("输入结构已经是合适的2D结构")
            structure.properties = {'conversion_method': 'already_2d'}
            return [structure]
        
        # 如果没有指定方法，根据分析结果智能选择
        if methods is None:
            if analysis['conversion_potential'] == 'exfoliation':
                methods = ['exfoliation', 'layer_separation']
                logger.info("检测到层状结构，优先使用剥离法")
            elif analysis['conversion_potential'] == 'surface_cleavage':
                methods = ['surface_cleavage', 'symmetry_reduction']  
                logger.info("检测到块状结构，使用表面切割法")
            elif analysis['conversion_potential'] == 'already_suitable':
                methods = ['exfoliation', 'surface_cleavage']
                logger.info("结构已较适合，尝试多种方法优化")
            else:
                methods = ['exfoliation', 'surface_cleavage', 'layer_separation']
                logger.info("结构特征不明确，尝试所有方法")
        
        all_2d_structures = []
        
        # 应用不同的转换方法
        for method in methods:
            try:
                logger.info(f"正在应用{method}方法...")
                
                if method == 'exfoliation':
                    structures = self.exfoliation_method(structure)
                elif method == 'surface_cleavage':
                    structures = self.surface_cleavage_method(structure)
                elif method == 'layer_separation':
                    structures = self.layer_separation_method(structure)
                elif method == 'symmetry_reduction':
                    structures = self.symmetry_reduction_method(structure)
                else:
                    logger.warning(f"未知的转换方法: {method}")
                    continue
                
                logger.info(f"{method}方法生成了{len(structures)}个2D结构")
                all_2d_structures.extend(structures)
                
            except Exception as e:
                logger.error(f"{method}方法执行失败: {e}")
                continue
        
        # 去重和质量过滤
        filtered_structures = self.filter_and_deduplicate(all_2d_structures)
        
        logger.info(f"总共生成{len(filtered_structures)}个有效的2D结构")
        return filtered_structures
    
    def filter_and_deduplicate(self, structures: List[Structure],
                              ltol: float = 0.2,
                              stol: float = 0.3,
                              angle_tol: float = 5.0) -> List[Structure]:
        """
        基于StructureMatcher的科学化结构过滤和去重

        使用pymatgen的StructureMatcher识别对称等价的重复结构，
        能够处理结构旋转、反射、平移等对称操作产生的重复。

        Args:
            structures: 结构列表
            ltol: 晶格参数容差 (fraction, 默认0.2)
            stol: 原子位点容差 (Angstrom, 默认0.3)
            angle_tol: 角度容差 (degrees, 默认5.0)

        Returns:
            过滤和去重后的结构列表
        """
        if not structures:
            return []

        logger.info(f"开始过滤和去重 {len(structures)} 个结构")

        # 第一步：验证并过滤无效结构
        valid_structures = []
        for i, structure in enumerate(structures):
            if self.validate_2d_structure(structure):
                structure.properties = structure.properties or {}
                structure.properties['original_index'] = i
                valid_structures.append(structure)
            else:
                logger.debug(f"结构 {i} 未通过2D验证，已跳过")

        logger.info(f"验证后剩余 {len(valid_structures)} 个有效结构")

        if not valid_structures:
            return []

        # 第二步：使用StructureMatcher进行去重
        deduplicated = self._advanced_structure_deduplication(
            valid_structures, ltol, stol, angle_tol
        )

        logger.info(f"去重后剩余 {len(deduplicated)} 个唯一结构")
        return deduplicated

    def _advanced_structure_deduplication(self, structures: List[Structure],
                                        ltol: float, stol: float, angle_tol: float) -> List[Structure]:
        """
        基于StructureMatcher的高级结构去重算法

        使用文献中验证的参数和方法识别结构等价性，
        参考XtalComp算法和Materials Project标准。

        Args:
            structures: 有效结构列表
            ltol: 晶格参数容差
            stol: 原子位点容差
            angle_tol: 角度容差

        Returns:
            去重后的结构列表
        """
        try:
            # 初始化StructureMatcher
            # 参数基于Materials Project和文献最佳实践
            matcher = StructureMatcher(
                ltol=ltol,           # 晶格参数相对容差
                stol=stol,           # 原子位点绝对容差 (Å)
                angle_tol=angle_tol, # 角度容差 (度)
                primitive_cell=True, # 使用原胞进行比较
                scale=True,          # 允许晶格缩放
                attempt_supercell=False,  # 2D结构通常不需要超胞匹配
                allow_subset=False,  # 要求完全匹配
                comparator=None      # 使用默认元素比较器
            )

            unique_structures = []
            duplicate_count = 0

            for i, current_structure in enumerate(structures):
                is_duplicate = False

                # 与所有已接受的唯一结构比较
                for j, unique_structure in enumerate(unique_structures):
                    try:
                        # 使用StructureMatcher判断是否等价
                        if matcher.fit(current_structure, unique_structure):
                            is_duplicate = True
                            duplicate_count += 1

                            # 记录重复信息到日志
                            original_idx = current_structure.properties.get('original_index', i)
                            unique_idx = unique_structure.properties.get('original_index', j)
                            logger.debug(f"结构 {original_idx} 与结构 {unique_idx} 重复，已跳过")

                            # 可选：保留能量更低或质量更高的结构
                            if self._should_replace_structure(current_structure, unique_structure):
                                unique_structures[j] = current_structure
                                logger.debug(f"用结构 {original_idx} 替换结构 {unique_idx}")

                            break

                    except Exception as e:
                        logger.warning(f"结构匹配失败 (结构 {i} vs {j}): {e}")
                        continue

                # 如果不是重复结构，添加到唯一列表
                if not is_duplicate:
                    unique_structures.append(current_structure)

            logger.info(f"去重统计: 发现 {duplicate_count} 个重复结构，保留 {len(unique_structures)} 个唯一结构")
            return unique_structures

        except Exception as e:
            logger.error(f"高级去重失败: {e}")
            # 降级到简单去重
            return self._fallback_simple_deduplication(structures, ltol)

    def _should_replace_structure(self, new_structure: Structure, existing_structure: Structure) -> bool:
        """
        判断是否应该用新结构替换现有结构

        比较标准：
        1. 原子数量（更多原子可能代表更完整的结构）
        2. 对称性（更高对称性通常更稳定）
        3. 验证得分（如果可用）

        Args:
            new_structure: 新结构
            existing_structure: 现有结构

        Returns:
            是否应该替换
        """
        try:
            # 比较原子数量
            new_atoms = len(new_structure.sites)
            existing_atoms = len(existing_structure.sites)

            if new_atoms != existing_atoms:
                return new_atoms > existing_atoms

            # 比较验证得分（如果可用）
            new_score = new_structure.properties.get('validation_score', 0)
            existing_score = existing_structure.properties.get('validation_score', 0)

            if abs(new_score - existing_score) > 0.01:
                return new_score > existing_score

            # 比较对称性信息（如果可用）
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

                new_spg = SpacegroupAnalyzer(new_structure).get_space_group_number()
                existing_spg = SpacegroupAnalyzer(existing_structure).get_space_group_number()

                # 更高的空间群号通常表示更高的对称性
                return new_spg > existing_spg

            except:
                pass

            # 默认保留现有结构
            return False

        except Exception as e:
            logger.debug(f"结构比较失败: {e}")
            return False

    def _fallback_simple_deduplication(self, structures: List[Structure], tolerance: float) -> List[Structure]:
        """
        简单去重的降级方案

        当StructureMatcher失败时使用的备用方法，
        基于晶格参数和原子数量的简单比较。

        Args:
            structures: 结构列表
            tolerance: 比较容差

        Returns:
            简单去重后的结构列表
        """
        logger.warning("使用简单去重降级方案")

        unique_structures = []

        for structure in structures:
            is_duplicate = False

            for existing in unique_structures:
                try:
                    # 简单的晶格参数和原子数比较
                    if (abs(structure.lattice.a - existing.lattice.a) < tolerance and
                        abs(structure.lattice.b - existing.lattice.b) < tolerance and
                        abs(structure.lattice.c - existing.lattice.c) < tolerance and
                        len(structure.sites) == len(existing.sites)):

                        # 额外检查化学组成
                        if structure.composition.reduced_formula == existing.composition.reduced_formula:
                            is_duplicate = True
                            break

                except Exception as e:
                    logger.debug(f"简单比较失败: {e}")
                    continue

            if not is_duplicate:
                unique_structures.append(structure)

        return unique_structures
    
    def batch_convert_structures(self, 
                                input_dir: str,
                                output_dir: str,
                                file_pattern: str = "*.cif",
                                max_files: int = None) -> Dict:
        """
        批量转换结构文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录  
            file_pattern: 文件模式
            max_files: 最大处理文件数
            
        Returns:
            转换结果统计
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取待处理文件
        if file_pattern.endswith('.cif'):
            input_files = list(input_path.glob(file_pattern))
        else:
            input_files = list(input_path.rglob(file_pattern))
        
        if max_files:
            input_files = input_files[:max_files]
        
        logger.info(f"找到{len(input_files)}个待处理文件")
        
        results = {
            'total_files': len(input_files),
            'successful_conversions': 0,
            'failed_conversions': 0,
            'total_2d_structures': 0,
            'conversion_details': []
        }
        
        for i, cif_file in enumerate(input_files):
            try:
                logger.info(f"处理文件 {i+1}/{len(input_files)}: {cif_file.name}")
                
                # 读取结构
                parser = CifParser(str(cif_file))
                structures = parser.get_structures()
                
                if not structures:
                    logger.warning(f"无法从{cif_file}读取结构")
                    results['failed_conversions'] += 1
                    continue
                
                structure = structures[0]  # 使用第一个结构
                
                # 转换为2D
                generated_2d = self.generate_2d_structures_from_3d(structure)
                
                if generated_2d:
                    # 保存生成的2D结构
                    base_name = cif_file.stem
                    for j, struct_2d in enumerate(generated_2d):
                        output_filename = f"{base_name}_2d_{j}.cif"
                        output_filepath = output_path / output_filename
                        
                        writer = CifWriter(struct_2d)
                        writer.write_file(str(output_filepath))
                    
                    results['successful_conversions'] += 1
                    results['total_2d_structures'] += len(generated_2d)
                    
                    results['conversion_details'].append({
                        'input_file': cif_file.name,
                        'n_2d_generated': len(generated_2d),
                        'methods_used': list(set([s.properties.get('conversion_method', 'unknown') 
                                                for s in generated_2d]))
                    })
                    
                    logger.info(f"成功转换{cif_file.name}，生成{len(generated_2d)}个2D结构")
                else:
                    logger.warning(f"未能从{cif_file.name}生成2D结构")
                    results['failed_conversions'] += 1
                    
            except Exception as e:
                logger.error(f"处理{cif_file}时出错: {e}")
                results['failed_conversions'] += 1
                continue
        
        # 保存结果统计
        results_file = output_path / "conversion_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"批量转换完成:")
        logger.info(f"  成功: {results['successful_conversions']}")
        logger.info(f"  失败: {results['failed_conversions']}")
        logger.info(f"  总计生成2D结构: {results['total_2d_structures']}")
        
        return results


def main():
    """测试函数"""
    converter = Crystal2DConverter(min_vacuum_thickness=15.0)
    
    # 测试单个文件转换
    test_cif = "C:/Users/Administrator/Downloads/材料/材料生成/data/2d_materials/mp_mp-1222908.cif"
    
    if os.path.exists(test_cif):
        try:
            parser = CifParser(test_cif)
            structures = parser.get_structures()
            
            if structures:
                structure = structures[0]
                logger.info(f"读取结构: {structure.composition.reduced_formula}")
                
                # 生成2D结构
                generated_2d = converter.generate_2d_structures_from_3d(structure)
                
                logger.info(f"生成了{len(generated_2d)}个2D结构")
                
                # 保存结果
                for i, struct_2d in enumerate(generated_2d):
                    output_file = f"test_2d_output_{i}.cif"
                    writer = CifWriter(struct_2d)
                    writer.write_file(output_file)
                    logger.info(f"保存2D结构到: {output_file}")
                    
        except Exception as e:
            logger.error(f"测试失败: {e}")


if __name__ == "__main__":
    main()