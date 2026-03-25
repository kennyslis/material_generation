"""
晶体结构标准化表示模块

该模块实现晶体结构的标准化表示，包括：
1. 晶体结构标准化处理
2. 多种特征表示方法（原子特征、晶体学特征、图表示）
3. 与MatterGen兼容的数据格式转换
4. 批量处理和缓存机制

作者：Claude
日期：2025-07-23
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
import json
import pickle
import logging
from tqdm import tqdm

from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import AutoOxidationStateDecorationTransformation
from pymatgen.io.cif import CifParser

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrystalFeatures:
    """晶体特征数据类"""
    # 基本信息
    material_id: str
    formula: str
    
    # 原子特征
    atomic_numbers: np.ndarray  # 原子序数 [N]
    atomic_positions: np.ndarray  # 原子坐标 [N, 3]
    num_atoms: int
    
    # 晶体学特征
    lattice_matrix: np.ndarray  # 晶格矩阵 [3, 3]
    lattice_parameters: np.ndarray  # a, b, c, alpha, beta, gamma [6]
    space_group: int
    crystal_system: str
    
    # 图表示特征
    adjacency_matrix: Optional[np.ndarray] = None  # 邻接矩阵 [N, N]
    edge_features: Optional[np.ndarray] = None  # 边特征 [E, F]
    node_features: Optional[np.ndarray] = None  # 节点特征 [N, F]
    
    # 材料属性
    properties: Dict = field(default_factory=dict)
    
    # 元数据
    metadata: Dict = field(default_factory=dict)


class CrystalStandardizer:
    """晶体结构标准化处理器"""
    
    def __init__(self, 
                 primitive: bool = True,
                 symprec: float = 0.1,
                 angle_tolerance: float = 5.0):
        """
        初始化标准化器
        
        Args:
            primitive: 是否转换为原胞
            symprec: 对称性精度
            angle_tolerance: 角度容差
        """
        self.primitive = primitive
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        
    def standardize(self, structure: Structure) -> Structure:
        """
        标准化晶体结构
        
        Args:
            structure: pymatgen Structure对象
            
        Returns:
            标准化后的Structure对象
        """
        # 1. 确保结构有序
        structure = structure.get_sorted_structure()
        
        # 2. 应用对称性分析
        sga = SpacegroupAnalyzer(structure, 
                                symprec=self.symprec,
                                angle_tolerance=self.angle_tolerance)
        
        # 3. 获取标准化结构
        if self.primitive:
            std_structure = sga.get_primitive_standard_structure()
        else:
            std_structure = sga.get_conventional_standard_structure()
            
        # 4. 确保晶格是右手系
        if std_structure.lattice.matrix.det() < 0:
            # 翻转一个轴以确保右手系
            new_matrix = std_structure.lattice.matrix.copy()
            new_matrix[2] = -new_matrix[2]
            std_structure.lattice = Lattice(new_matrix)
            
        return std_structure


class CrystalFeatureExtractor:
    """晶体特征提取器"""
    
    def __init__(self,
                 standardizer: Optional[CrystalStandardizer] = None,
                 compute_graph: bool = True,
                 cutoff_radius: float = 8.0):
        """
        初始化特征提取器
        
        Args:
            standardizer: 晶体标准化器
            compute_graph: 是否计算图特征
            cutoff_radius: 图构建的截断半径
        """
        self.standardizer = standardizer or CrystalStandardizer()
        self.compute_graph = compute_graph
        self.cutoff_radius = cutoff_radius
        
        # 元素特征（原子序数到特征的映射）
        self._init_element_features()
        
    def _init_element_features(self):
        """初始化元素特征表"""
        # 这里使用简化的元素特征，实际可以使用更复杂的特征
        self.element_features = {}
        
        # 基本元素属性：[电负性, 原子半径, 价电子数, 周期, 族]
        basic_features = {
            'H': [2.20, 0.31, 1, 1, 1],
            'C': [2.55, 0.76, 4, 2, 14],
            'N': [3.04, 0.71, 5, 2, 15],
            'O': [3.44, 0.66, 6, 2, 16],
            'Si': [1.90, 1.11, 4, 3, 14],
            'S': [2.58, 1.05, 6, 3, 16],
            # 添加更多元素...
        }
        
        # 标准化特征
        for symbol, features in basic_features.items():
            self.element_features[symbol] = np.array(features, dtype=np.float32)
            
    def extract_features(self, structure: Structure, material_id: str = None) -> CrystalFeatures:
        """
        提取晶体结构特征
        
        Args:
            structure: pymatgen Structure对象
            material_id: 材料ID
            
        Returns:
            CrystalFeatures对象
        """
        # 标准化结构
        std_structure = self.standardizer.standardize(structure)
        
        # 获取对称性信息
        sga = SpacegroupAnalyzer(std_structure)
        space_group = sga.get_space_group_number()
        crystal_system = sga.get_crystal_system()
        
        # 提取原子信息
        atomic_numbers = np.array([site.specie.Z for site in std_structure])
        atomic_positions = std_structure.frac_coords.astype(np.float32)
        
        # 晶格参数
        lattice = std_structure.lattice
        lattice_matrix = lattice.matrix.astype(np.float32)
        lattice_parameters = np.array([
            lattice.a, lattice.b, lattice.c,
            lattice.alpha, lattice.beta, lattice.gamma
        ], dtype=np.float32)
        
        # 构建特征对象
        features = CrystalFeatures(
            material_id=material_id or std_structure.composition.reduced_formula,
            formula=std_structure.composition.reduced_formula,
            atomic_numbers=atomic_numbers,
            atomic_positions=atomic_positions,
            num_atoms=len(std_structure),
            lattice_matrix=lattice_matrix,
            lattice_parameters=lattice_parameters,
            space_group=space_group,
            crystal_system=crystal_system,
            properties={
                'volume': std_structure.volume,
                'density': std_structure.density,
            },
            metadata={
                'standardized': True,
                'primitive': self.standardizer.primitive,
            }
        )
        
        # 计算图特征
        if self.compute_graph:
            self._compute_graph_features(std_structure, features)
            
        # 计算节点特征
        self._compute_node_features(std_structure, features)
        
        return features
        
    def _compute_graph_features(self, structure: Structure, features: CrystalFeatures):
        """计算图表示特征"""
        n_atoms = len(structure)
        adjacency_matrix = np.zeros((n_atoms, n_atoms), dtype=np.float32)
        edge_features_list = []
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    dist = structure.get_distance(i, j)
                    if dist < self.cutoff_radius:
                        adjacency_matrix[i, j] = 1.0
                        # 边特征：[距离, 1/距离, exp(-距离)]
                        edge_feat = np.array([
                            dist,
                            1.0 / (dist + 1e-8),
                            np.exp(-dist)
                        ], dtype=np.float32)
                        edge_features_list.append(edge_feat)
                        
        features.adjacency_matrix = adjacency_matrix
        if edge_features_list:
            features.edge_features = np.array(edge_features_list)
            
    def _compute_node_features(self, structure: Structure, features: CrystalFeatures):
        """计算节点（原子）特征"""
        node_features_list = []
        
        for site in structure:
            # 获取元素符号
            element = site.specie.symbol
            
            # 基础特征
            if element in self.element_features:
                elem_feat = self.element_features[element]
            else:
                # 使用默认特征
                elem_feat = np.array([1.0, 1.0, 4.0, 3.0, 14.0], dtype=np.float32)
                
            # 添加配位环境特征（简化版）
            neighbors = structure.get_neighbors(site, self.cutoff_radius)
            coord_number = len(neighbors)
            avg_neighbor_dist = np.mean([n[1] for n in neighbors]) if neighbors else 0.0
            
            # 组合特征
            node_feat = np.concatenate([
                elem_feat,
                [coord_number, avg_neighbor_dist]
            ])
            
            node_features_list.append(node_feat)
            
        features.node_features = np.array(node_features_list, dtype=np.float32)


class CrystalRepresentationConverter:
    """晶体表示格式转换器"""
    
    def __init__(self, feature_extractor: Optional[CrystalFeatureExtractor] = None):
        """
        初始化转换器
        
        Args:
            feature_extractor: 特征提取器
        """
        self.feature_extractor = feature_extractor or CrystalFeatureExtractor()
        
    def to_mattergen_format(self, features: CrystalFeatures) -> Dict:
        """
        转换为MatterGen兼容格式
        
        Args:
            features: CrystalFeatures对象
            
        Returns:
            MatterGen格式的字典
        """
        mattergen_data = {
            'material_id': features.material_id,
            'formula': features.formula,
            'num_atoms': features.num_atoms,
            
            # 原子数据
            'atomic_numbers': features.atomic_numbers.tolist(),
            'frac_coords': features.atomic_positions.tolist(),
            
            # 晶格数据
            'lattice': {
                'matrix': features.lattice_matrix.tolist(),
                'a': float(features.lattice_parameters[0]),
                'b': float(features.lattice_parameters[1]),
                'c': float(features.lattice_parameters[2]),
                'alpha': float(features.lattice_parameters[3]),
                'beta': float(features.lattice_parameters[4]),
                'gamma': float(features.lattice_parameters[5]),
            },
            
            # 对称性
            'spacegroup': {
                'number': features.space_group,
                'crystal_system': features.crystal_system,
            },
            
            # 属性
            'properties': features.properties,
            
            # 图数据（如果有）
            'graph_data': {}
        }
        
        # 添加图数据
        if features.adjacency_matrix is not None:
            # 转换为边列表格式
            edges = np.argwhere(features.adjacency_matrix > 0)
            mattergen_data['graph_data']['edges'] = edges.tolist()
            
        if features.node_features is not None:
            mattergen_data['graph_data']['node_features'] = features.node_features.tolist()
            
        if features.edge_features is not None:
            mattergen_data['graph_data']['edge_features'] = features.edge_features.tolist()
            
        return mattergen_data
        
    def from_cif_file(self, cif_path: str, material_id: Optional[str] = None) -> CrystalFeatures:
        """
        从CIF文件提取特征
        
        Args:
            cif_path: CIF文件路径
            material_id: 材料ID
            
        Returns:
            CrystalFeatures对象
        """
        parser = CifParser(cif_path)
        structure = parser.get_structures()[0]
        
        if material_id is None:
            material_id = Path(cif_path).stem
            
        return self.feature_extractor.extract_features(structure, material_id)
        
    def batch_process_materials(self, 
                              csv_path: str,
                              output_dir: str,
                              batch_size: int = 100) -> None:
        """
        批量处理材料数据
        
        Args:
            csv_path: 材料CSV文件路径
            output_dir: 输出目录
            batch_size: 批处理大小
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 读取材料列表
        df = pd.read_csv(csv_path)
        logger.info(f"加载了 {len(df)} 个材料")
        
        # 批量处理
        all_features = []
        failed_materials = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理材料"):
            try:
                # 提取特征
                cif_path = row['cif_file']
                material_id = row['material_id']
                
                features = self.from_cif_file(cif_path, material_id)
                
                # 添加额外属性
                if 'energy_above_hull' in row:
                    features.properties['energy_above_hull'] = row['energy_above_hull']
                if 'formation_energy_per_atom' in row:
                    features.properties['formation_energy_per_atom'] = row['formation_energy_per_atom']
                
                all_features.append(features)
                
                # 定期保存
                if len(all_features) % batch_size == 0:
                    self._save_batch(all_features[-batch_size:], output_path, idx)
                    
            except Exception as e:
                logger.warning(f"处理材料 {material_id} 失败: {e}")
                failed_materials.append((material_id, str(e)))
                
        # 保存剩余的特征
        if len(all_features) % batch_size != 0:
            remaining = len(all_features) % batch_size
            self._save_batch(all_features[-remaining:], output_path, len(df))
            
        # 保存失败记录
        if failed_materials:
            with open(output_path / 'failed_materials.json', 'w') as f:
                json.dump(failed_materials, f, indent=2)
                
        logger.info(f"处理完成！成功: {len(all_features)}, 失败: {len(failed_materials)}")
        
    def _save_batch(self, features_batch: List[CrystalFeatures], output_dir: Path, batch_idx: int):
        """保存一批特征"""
        # 保存为pickle格式（保留numpy数组）
        pickle_path = output_dir / f'features_batch_{batch_idx:05d}.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(features_batch, f)
            
        # 同时保存为MatterGen格式
        mattergen_batch = []
        for features in features_batch:
            mattergen_data = self.to_mattergen_format(features)
            mattergen_batch.append(mattergen_data)
            
        json_path = output_dir / f'mattergen_batch_{batch_idx:05d}.json'
        with open(json_path, 'w') as f:
            json.dump(mattergen_batch, f, indent=2)


def test_crystal_representation():
    """测试晶体表示功能"""
    # 创建测试结构（石墨烯）
    a = 2.46
    lattice = Lattice.hexagonal(a, 20)
    species = ['C', 'C']
    coords = [[0, 0, 0.5], [1/3, 2/3, 0.5]]
    graphene = Structure(lattice, species, coords)
    
    # 创建特征提取器
    extractor = CrystalFeatureExtractor()
    
    # 提取特征
    features = extractor.extract_features(graphene, "test_graphene")
    
    print("晶体特征提取测试")
    print("=" * 50)
    print(f"材料ID: {features.material_id}")
    print(f"化学式: {features.formula}")
    print(f"原子数: {features.num_atoms}")
    print(f"空间群: {features.space_group}")
    print(f"晶系: {features.crystal_system}")
    print(f"晶格参数: {features.lattice_parameters}")
    print(f"体积: {features.properties['volume']:.2f} Å³")
    print(f"密度: {features.properties['density']:.2f} g/cm³")
    
    if features.node_features is not None:
        print(f"节点特征形状: {features.node_features.shape}")
        
    # 测试格式转换
    converter = CrystalRepresentationConverter(extractor)
    mattergen_data = converter.to_mattergen_format(features)
    
    print("\nMatterGen格式转换测试")
    print(f"键数量: {len(mattergen_data.keys())}")
    print(f"包含图数据: {'edges' in mattergen_data.get('graph_data', {})}")
    
    return features, mattergen_data


if __name__ == "__main__":
    test_crystal_representation()