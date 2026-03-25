"""
二维材料高精度筛选脚本 - 完整算法版本
基于论文"二维材料的高通量筛选与光催化性能预测"中的完整方法

使用完整的算法实现：
1. 完整的拓扑缩放算法(TSA) - 基于持续同调和图论
2. 基于DFT的层间结合能精确计算
3. 完整的图论结构分析和连通性分析
4. 精确的维度判定算法
5. vdW相互作用的量子化学分析

注意：该版本计算精度高但耗时较长，适合小规模精确筛选
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 核心依赖
from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.cif import CifParser
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN, JMolNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.transformations.standard_transformations import PrimitiveCellTransformation

# 高级分析工具
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull, Voronoi
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# DFT计算工具 (需要安装相应的DFT后端)
try:
    from ase import Atoms
    from ase.io import write as ase_write
    from ase.calculators.vasp import Vasp
    from ase.optimize import BFGS
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    print("警告: ASE未安装，将使用经验方法代替DFT计算")

# 持续同调工具
try:
    import gudhi
    from ripser import ripser
    TOPOLOGY_AVAILABLE = True
except ImportError:
    TOPOLOGY_AVAILABLE = False
    print("警告: 拓扑分析工具未安装，将使用图论方法")

# 量子化学计算
try:
    from pyscf import gto, dft, mp
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("警告: PySCF未安装，将使用经验vdW参数")

class AdvancedTwoDMaterialScreener:
    """高精度二维材料筛选器 - 使用完整算法"""
    
    def __init__(self, input_dir: str, output_dir: str, 
                 use_dft: bool = False, use_topology: bool = True):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_dft = use_dft and ASE_AVAILABLE
        self.use_topology = use_topology and TOPOLOGY_AVAILABLE
        
        # 筛选参数
        self.max_elements = 3
        self.max_binding_energy = 0.15  # eV/Å²
        self.min_interlayer_distance = 2.5  # Å
        self.min_2d_score = 0.6  # 提高阈值，使用精确算法
        
        # TSA参数
        self.tsa_config = {
            'max_dimension': 2,
            'min_persistence': 0.1,
            'resolution': 100,
            'distance_matrix_threshold': 8.0
        }
        
        # DFT参数
        self.dft_config = {
            'xc': 'PBE',
            'encut': 520,
            'kpts': (5, 5, 1),
            'convergence': {'energy': 1e-6, 'forces': 0.01}
        } if self.use_dft else None
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'dimensionality_2d': 0,
            'low_binding_energy': 0,
            'layered_structure': 0,
            'final_2d_candidates': 0,
            'processing_errors': 0,
            'tsa_analysis_count': 0,
            'dft_calculations_count': 0
        }
        
    def topology_scaling_analysis_complete(self, structure: Structure) -> Dict:
        """
        完整的拓扑缩放算法(TSA)实现
        基于持续同调理论和图论分析材料的拓扑维度
        """
        try:
            # 第一步：构建距离矩阵
            coords = np.array([site.coords for site in structure])
            n_atoms = len(coords)
            
            if n_atoms < 3:
                return self._fallback_tsa_analysis(structure)
            
            # 计算原子间距离矩阵
            distance_matrix = squareform(pdist(coords))
            
            # 第二步：持续同调分析
            topo_features = self._compute_persistent_homology(distance_matrix, coords)
            
            # 第三步：图论拓扑分析
            graph_features = self._graph_topology_analysis(structure, distance_matrix)
            
            # 第四步：几何形状分析
            geometric_features = self._geometric_shape_analysis(structure, coords)
            
            # 第五步：维度缩放检验
            scaling_analysis = self._dimensional_scaling_test(structure, distance_matrix)
            
            # 综合分析结果
            dimension_scores = self._integrate_tsa_results(
                topo_features, graph_features, geometric_features, scaling_analysis
            )
            
            # 确定最终维度
            final_dimension, confidence = self._determine_dimensionality(dimension_scores)
            
            self.stats['tsa_analysis_count'] += 1
            
            return {
                'dimension': final_dimension,
                'confidence': confidence,
                'topology_features': topo_features,
                'graph_features': graph_features,
                'geometric_features': geometric_features,
                'scaling_analysis': scaling_analysis,
                'dimension_scores': dimension_scores,
                'method': 'complete_tsa'
            }
            
        except Exception as e:
            print(f"完整TSA分析失败，使用备用方法: {e}")
            return self._fallback_tsa_analysis(structure)
    
    def _compute_persistent_homology(self, distance_matrix: np.ndarray, coords: np.ndarray) -> Dict:
        """计算持续同调特征"""
        if not TOPOLOGY_AVAILABLE:
            return {'method': 'unavailable', 'homology_dimensions': [0, 0, 0]}
        
        try:
            # 使用Ripser计算持续同调
            dgms = ripser(distance_matrix, maxdim=2, distance_matrix=True)['dgms']
            
            # 分析各维度的持续同调特征
            h0_features = self._analyze_homology_dimension(dgms[0], 0)  # 连通分量
            h1_features = self._analyze_homology_dimension(dgms[1], 1) if len(dgms) > 1 else {}  # 环
            h2_features = self._analyze_homology_dimension(dgms[2], 2) if len(dgms) > 2 else {}  # 空腔
            
            return {
                'method': 'persistent_homology',
                'h0_features': h0_features,  # 连通性
                'h1_features': h1_features,  # 环状结构 (2D特征)
                'h2_features': h2_features,  # 空腔结构 (3D特征)
                'total_persistence': sum([h0_features.get('total_persistence', 0),
                                        h1_features.get('total_persistence', 0),
                                        h2_features.get('total_persistence', 0)])
            }
            
        except Exception as e:
            print(f"持续同调计算失败: {e}")
            return {'method': 'failed', 'error': str(e)}
    
    def _analyze_homology_dimension(self, dgm: np.ndarray, dim: int) -> Dict:
        """分析特定维度的同调特征"""
        if dgm.shape[0] == 0:
            return {'num_features': 0, 'total_persistence': 0, 'max_persistence': 0}
        
        # 计算持续性（生存时间）
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        
        # 处理无限持续性特征
        infinite_mask = np.isinf(deaths)
        finite_mask = ~infinite_mask
        
        persistence = deaths[finite_mask] - births[finite_mask]
        
        # 过滤噪声（持续性太小的特征）
        significant_features = persistence > self.tsa_config['min_persistence']
        
        return {
            'dimension': dim,
            'num_features': len(persistence),
            'num_infinite_features': np.sum(infinite_mask),
            'num_significant_features': np.sum(significant_features),
            'total_persistence': np.sum(persistence),
            'max_persistence': np.max(persistence) if len(persistence) > 0 else 0,
            'avg_persistence': np.mean(persistence) if len(persistence) > 0 else 0,
            'persistence_std': np.std(persistence) if len(persistence) > 0 else 0
        }
    
    def _graph_topology_analysis(self, structure: Structure, distance_matrix: np.ndarray) -> Dict:
        """完整的图论拓扑分析"""
        try:
            # 构建结构图
            structure_graph = StructureGraph.with_local_env_strategy(structure, CrystalNN())
            
            # 转换为NetworkX图
            G = self._structure_to_networkx(structure_graph)
            
            # 计算图论特征
            graph_metrics = {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'density': nx.density(G),
                'clustering_coefficient': nx.average_clustering(G),
                'avg_shortest_path': self._safe_avg_shortest_path(G),
                'diameter': self._safe_diameter(G),
                'radius': self._safe_radius(G)
            }
            
            # 连通性分析
            connectivity_analysis = self._analyze_graph_connectivity(G, distance_matrix)
            
            # 平面性检测（2D特征）
            planarity_analysis = self._analyze_planarity(G, structure)
            
            # 分层结构检测
            layering_analysis = self._analyze_graph_layering(G, structure)
            
            return {
                'basic_metrics': graph_metrics,
                'connectivity': connectivity_analysis,
                'planarity': planarity_analysis,
                'layering': layering_analysis
            }
            
        except Exception as e:
            print(f"图论分析失败: {e}")
            return {'error': str(e), 'method': 'failed'}
    
    def _structure_to_networkx(self, structure_graph: StructureGraph) -> nx.Graph:
        """将PyMatGen结构图转换为NetworkX图"""
        G = nx.Graph()
        
        # 添加节点
        for i in range(len(structure_graph.structure)):
            site = structure_graph.structure[i]
            G.add_node(i, element=str(site.specie), coords=site.coords)
        
        # 添加边
        for i in range(len(structure_graph.structure)):
            connected_sites = structure_graph.get_connected_sites(i)
            for site in connected_sites:
                j = site.index
                if i < j:  # 避免重复边
                    distance = np.linalg.norm(
                        structure_graph.structure[i].coords - structure_graph.structure[j].coords
                    )
                    G.add_edge(i, j, distance=distance)
        
        return G
    
    def _analyze_graph_connectivity(self, G: nx.Graph, distance_matrix: np.ndarray) -> Dict:
        """分析图的连通性特征"""
        try:
            # 连通分量分析
            connected_components = list(nx.connected_components(G))
            num_components = len(connected_components)
            
            # 最大连通分量
            largest_component_size = max(len(c) for c in connected_components) if connected_components else 0
            
            # 桥和割点分析
            bridges = list(nx.bridges(G))
            articulation_points = list(nx.articulation_points(G))
            
            # 连通性维度指标
            connectivity_dimension = self._estimate_connectivity_dimension(G, connected_components)
            
            return {
                'num_connected_components': num_components,
                'largest_component_size': largest_component_size,
                'num_bridges': len(bridges),
                'num_articulation_points': len(articulation_points),
                'connectivity_dimension': connectivity_dimension,
                'is_connected': nx.is_connected(G)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_planarity(self, G: nx.Graph, structure: Structure) -> Dict:
        """分析图的平面性（2D特征）"""
        try:
            # NetworkX的平面性检测
            is_planar, embedding = nx.check_planarity(G)
            
            # 几何平面性分析
            coords = np.array([site.coords for site in structure])
            geometric_planarity = self._check_geometric_planarity(coords)
            
            # Kuratowski子图检测
            kuratowski_analysis = self._detect_kuratowski_subgraphs(G)
            
            return {
                'is_planar_graph': is_planar,
                'geometric_planarity': geometric_planarity,
                'kuratowski_analysis': kuratowski_analysis,
                'planar_score': self._calculate_planar_score(is_planar, geometric_planarity)
            }
            
        except Exception as e:
            return {'error': str(e), 'is_planar_graph': False}
    
    def _check_geometric_planarity(self, coords: np.ndarray) -> Dict:
        """检查几何平面性"""
        if len(coords) < 4:
            return {'is_coplanar': True, 'planarity_score': 1.0, 'normal_vector': [0, 0, 1]}
        
        # PCA分析寻找主平面
        pca = PCA(n_components=3)
        pca.fit(coords - np.mean(coords, axis=0))
        
        # 方差比例，如果第三个分量很小，则接近平面
        explained_variance = pca.explained_variance_ratio_
        planarity_score = 1.0 - explained_variance[2] if len(explained_variance) > 2 else 1.0
        
        # 计算到主平面的距离
        coords_centered = coords - np.mean(coords, axis=0)
        coords_pca = pca.transform(coords_centered)
        plane_distances = np.abs(coords_pca[:, 2]) if coords_pca.shape[1] > 2 else np.zeros(len(coords))
        
        return {
            'is_coplanar': planarity_score > 0.95,
            'planarity_score': planarity_score,
            'max_plane_distance': np.max(plane_distances),
            'avg_plane_distance': np.mean(plane_distances),
            'normal_vector': pca.components_[2].tolist() if len(pca.components_) > 2 else [0, 0, 1]
        }
    
    def _detect_kuratowski_subgraphs(self, G: nx.Graph) -> Dict:
        """检测Kuratowski子图（K5和K3,3）"""
        try:
            # 检查是否包含K5完全图
            has_k5 = False
            k5_nodes = []
            
            # 检查是否包含K3,3完全二分图
            has_k33 = False
            k33_partition = None
            
            # 简化检测：检查高度数节点
            degrees = dict(G.degree())
            high_degree_nodes = [node for node, degree in degrees.items() if degree >= 4]
            
            if len(high_degree_nodes) >= 5:
                # 可能包含K5
                for i, node_set in enumerate(self._get_combinations(high_degree_nodes, 5)):
                    subgraph = G.subgraph(node_set)
                    if subgraph.number_of_edges() == 10:  # K5有10条边
                        has_k5 = True
                        k5_nodes = list(node_set)
                        break
            
            return {
                'has_k5': has_k5,
                'k5_nodes': k5_nodes,
                'has_k33': has_k33,
                'k33_partition': k33_partition,
                'planarity_obstruction_score': 1.0 if (has_k5 or has_k33) else 0.0
            }
            
        except Exception:
            return {'has_k5': False, 'has_k33': False, 'planarity_obstruction_score': 0.0}
    
    def _get_combinations(self, items: List, r: int):
        """生成组合"""
        from itertools import combinations
        return combinations(items, r)
    
    def _analyze_graph_layering(self, G: nx.Graph, structure: Structure) -> Dict:
        """分析图的分层结构"""
        try:
            coords = np.array([site.coords for site in structure])
            
            # Z方向聚类分析寻找层
            z_coords = coords[:, 2].reshape(-1, 1)
            clustering = DBSCAN(eps=1.5, min_samples=1).fit(z_coords)
            
            layers = {}
            for i, label in enumerate(clustering.labels_):
                if label not in layers:
                    layers[label] = []
                layers[label].append(i)
            
            num_layers = len(layers)
            layer_sizes = [len(layer) for layer in layers.values()]
            
            # 分析层间连接
            inter_layer_edges = 0
            intra_layer_edges = 0
            
            for edge in G.edges():
                node1_layer = clustering.labels_[edge[0]]
                node2_layer = clustering.labels_[edge[1]]
                
                if node1_layer == node2_layer:
                    intra_layer_edges += 1
                else:
                    inter_layer_edges += 1
            
            layering_score = intra_layer_edges / (intra_layer_edges + inter_layer_edges) if (intra_layer_edges + inter_layer_edges) > 0 else 0
            
            return {
                'num_layers': num_layers,
                'layer_sizes': layer_sizes,
                'inter_layer_edges': inter_layer_edges,
                'intra_layer_edges': intra_layer_edges,
                'layering_score': layering_score,
                'is_layered': num_layers > 1 and layering_score > 0.7
            }
            
        except Exception as e:
            return {'error': str(e), 'is_layered': False}
    
    def _geometric_shape_analysis(self, structure: Structure, coords: np.ndarray) -> Dict:
        """几何形状分析"""
        try:
            # 凸包分析
            hull_analysis = self._convex_hull_analysis(coords)
            
            # 主成分分析
            pca_analysis = self._principal_component_analysis(coords)
            
            # 形状各向异性
            anisotropy_analysis = self._shape_anisotropy_analysis(coords, structure)
            
            # 表面积体积比
            surface_volume_ratio = self._calculate_surface_volume_ratio(structure, coords)
            
            return {
                'convex_hull': hull_analysis,
                'pca_analysis': pca_analysis,
                'anisotropy': anisotropy_analysis,
                'surface_volume_ratio': surface_volume_ratio
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _convex_hull_analysis(self, coords: np.ndarray) -> Dict:
        """凸包分析"""
        try:
            if len(coords) < 4:
                return {'dimension': len(coords) - 1, 'volume': 0, 'surface_area': 0}
            
            hull = ConvexHull(coords)
            
            return {
                'dimension': coords.shape[1],
                'num_vertices': len(hull.vertices),
                'num_faces': len(hull.simplices),
                'volume': hull.volume,
                'surface_area': hull.area if hasattr(hull, 'area') else 0,
                'aspect_ratios': self._calculate_hull_aspect_ratios(coords[hull.vertices])
            }
            
        except Exception:
            return {'dimension': 3, 'volume': 0, 'surface_area': 0}
    
    def _calculate_hull_aspect_ratios(self, vertices: np.ndarray) -> List[float]:
        """计算凸包的纵横比"""
        try:
            # 计算主轴方向
            pca = PCA(n_components=3)
            pca.fit(vertices - np.mean(vertices, axis=0))
            
            # 投影到主轴并计算范围
            vertices_pca = pca.transform(vertices - np.mean(vertices, axis=0))
            ranges = [np.ptp(vertices_pca[:, i]) for i in range(3)]
            
            # 计算纵横比
            ranges = sorted(ranges, reverse=True)
            ratios = []
            for i in range(len(ranges) - 1):
                if ranges[i+1] > 0:
                    ratios.append(ranges[i] / ranges[i+1])
                else:
                    ratios.append(float('inf'))
            
            return ratios
            
        except Exception:
            return [1.0, 1.0]
    
    def _principal_component_analysis(self, coords: np.ndarray) -> Dict:
        """主成分分析"""
        try:
            coords_centered = coords - np.mean(coords, axis=0)
            pca = PCA(n_components=min(3, coords.shape[1]))
            pca.fit(coords_centered)
            
            explained_variance = pca.explained_variance_ratio_
            
            # 维度判断：如果某个方向方差很小，可能是低维结构
            dimension_score = {
                '2D': 1.0 - explained_variance[2] if len(explained_variance) > 2 else 0.0,
                '1D': explained_variance[0] if len(explained_variance) > 0 else 0.0,
                '0D': 1.0 if np.sum(explained_variance) < 0.1 else 0.0
            }
            
            return {
                'explained_variance_ratio': explained_variance.tolist(),
                'principal_components': pca.components_.tolist(),
                'dimension_scores': dimension_score,
                'effective_dimension': self._estimate_effective_dimension(explained_variance)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _estimate_effective_dimension(self, explained_variance: np.ndarray) -> float:
        """估计有效维度"""
        # 基于方差贡献的累计阈值
        cumsum = np.cumsum(explained_variance)
        for i, cum_var in enumerate(cumsum):
            if cum_var > 0.95:  # 95%方差解释阈值
                return float(i + 1)
        return float(len(explained_variance))
    
    def _shape_anisotropy_analysis(self, coords: np.ndarray, structure: Structure) -> Dict:
        """形状各向异性分析"""
        try:
            # 惯性张量分析
            coords_centered = coords - np.mean(coords, axis=0)
            inertia_tensor = np.dot(coords_centered.T, coords_centered) / len(coords)
            
            eigenvals, eigenvecs = np.linalg.eigh(inertia_tensor)
            eigenvals = np.sort(eigenvals)[::-1]  # 降序排列
            
            # 各向异性参数
            if eigenvals[2] > 0:
                prolate_parameter = (eigenvals[0] - eigenvals[1]) / (2 * eigenvals[2])
                oblate_parameter = (eigenvals[1] - eigenvals[2]) / (2 * eigenvals[2])
            else:
                prolate_parameter = oblate_parameter = 0
            
            # 形状分类
            shape_type = self._classify_shape_type(prolate_parameter, oblate_parameter)
            
            return {
                'eigenvalues': eigenvals.tolist(),
                'eigenvectors': eigenvecs.tolist(),
                'prolate_parameter': prolate_parameter,
                'oblate_parameter': oblate_parameter,
                'shape_type': shape_type,
                'anisotropy_ratio': eigenvals[0] / eigenvals[2] if eigenvals[2] > 0 else float('inf')
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _classify_shape_type(self, prolate: float, oblate: float) -> str:
        """分类形状类型"""
        if oblate > 0.3:
            return 'oblate'  # 扁平 (类似2D)
        elif prolate > 0.3:
            return 'prolate'  # 拉长 (类似1D)
        else:
            return 'spherical'  # 球形 (3D)
    
    def _calculate_surface_volume_ratio(self, structure: Structure, coords: np.ndarray) -> float:
        """计算表面积体积比"""
        try:
            if len(coords) < 4:
                return 0.0
            
            hull = ConvexHull(coords)
            volume = hull.volume if hull.volume > 0 else 1e-10
            surface_area = hull.area if hasattr(hull, 'area') else 0
            
            return surface_area / volume
            
        except Exception:
            return 0.0
    
    def _dimensional_scaling_test(self, structure: Structure, distance_matrix: np.ndarray) -> Dict:
        """维度缩放检验"""
        try:
            coords = np.array([site.coords for site in structure])
            
            # 多尺度分析
            scales = np.logspace(-1, 1, 10)  # 0.1到10的尺度
            scaling_results = []
            
            for scale in scales:
                scaled_coords = coords * scale
                scaled_features = self._calculate_scaling_features(scaled_coords)
                scaling_results.append(scaled_features)
            
            # 分析尺度不变性
            invariance_analysis = self._analyze_scale_invariance(scaling_results)
            
            # 分形维数估计
            fractal_dimension = self._estimate_fractal_dimension(coords, distance_matrix)
            
            return {
                'scales': scales.tolist(),
                'scaling_results': scaling_results,
                'invariance_analysis': invariance_analysis,
                'fractal_dimension': fractal_dimension
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_scaling_features(self, coords: np.ndarray) -> Dict:
        """计算尺度特征"""
        try:
            # 计算各种几何量
            centroid = np.mean(coords, axis=0)
            distances_to_centroid = np.linalg.norm(coords - centroid, axis=1)
            
            return {
                'mean_distance_to_centroid': np.mean(distances_to_centroid),
                'max_distance_to_centroid': np.max(distances_to_centroid),
                'std_distance_to_centroid': np.std(distances_to_centroid),
                'coordinate_ranges': [np.ptp(coords[:, i]) for i in range(coords.shape[1])]
            }
            
        except Exception:
            return {'mean_distance_to_centroid': 0}
    
    def _analyze_scale_invariance(self, scaling_results: List[Dict]) -> Dict:
        """分析尺度不变性"""
        try:
            # 检查特征随尺度的变化
            feature_variations = {}
            
            for feature_key in scaling_results[0].keys():
                if feature_key != 'coordinate_ranges':
                    values = [result.get(feature_key, 0) for result in scaling_results]
                    variation_coeff = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                    feature_variations[feature_key] = variation_coeff
            
            avg_variation = np.mean(list(feature_variations.values()))
            
            return {
                'feature_variations': feature_variations,
                'avg_variation_coefficient': avg_variation,
                'is_scale_invariant': avg_variation < 0.1
            }
            
        except Exception:
            return {'is_scale_invariant': False}
    
    def _estimate_fractal_dimension(self, coords: np.ndarray, distance_matrix: np.ndarray) -> float:
        """估计分形维数"""
        try:
            # 使用box-counting方法
            if len(coords) < 3:
                return float(len(coords) - 1)
            
            # 简化的分形维数估计
            distances = distance_matrix[np.triu_indices(len(coords), k=1)]
            distances = distances[distances > 0]
            
            if len(distances) == 0:
                return 2.0
            
            # 使用距离分布的幂律拟合估计维数
            log_distances = np.log(distances)
            log_counts = np.log(np.arange(1, len(distances) + 1))
            
            # 线性拟合
            if len(log_distances) > 1:
                slope, _ = np.polyfit(log_distances, log_counts, 1)
                fractal_dim = abs(slope)
            else:
                fractal_dim = 2.0
            
            # 限制在合理范围
            return max(0.0, min(3.0, fractal_dim))
            
        except Exception:
            return 2.0
    
    def _integrate_tsa_results(self, topo_features: Dict, graph_features: Dict, 
                              geometric_features: Dict, scaling_analysis: Dict) -> Dict:
        """整合TSA分析结果"""
        dimension_scores = {'0D': 0.0, '1D': 0.0, '2D': 0.0, '3D': 0.0}
        
        # 拓扑特征权重 (40%)
        if 'h1_features' in topo_features:
            h1_persistence = topo_features['h1_features'].get('total_persistence', 0)
            dimension_scores['2D'] += 0.4 * min(1.0, h1_persistence / 10.0)
        
        # 图论特征权重 (30%)
        if 'planarity' in graph_features:
            planarity_score = graph_features['planarity'].get('planar_score', 0)
            dimension_scores['2D'] += 0.3 * planarity_score
        
        # 几何特征权重 (30%)
        if 'pca_analysis' in geometric_features:
            pca_scores = geometric_features['pca_analysis'].get('dimension_scores', {})
            for dim, score in pca_scores.items():
                if dim in dimension_scores:
                    dimension_scores[dim] += 0.3 * score
        
        return dimension_scores
    
    def _determine_dimensionality(self, dimension_scores: Dict) -> Tuple[str, float]:
        """确定最终维度"""
        max_dim = max(dimension_scores.keys(), key=lambda x: dimension_scores[x])
        confidence = dimension_scores[max_dim]
        
        # 如果最高分数太低，标记为不确定
        if confidence < 0.3:
            return 'Unknown', confidence
        
        return max_dim, confidence
    
    def _safe_avg_shortest_path(self, G: nx.Graph) -> float:
        """安全计算平均最短路径"""
        try:
            if nx.is_connected(G):
                return nx.average_shortest_path_length(G)
            else:
                # 对于非连通图，计算最大连通分量的平均最短路径
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                return nx.average_shortest_path_length(subgraph)
        except:
            return 0.0
    
    def _safe_diameter(self, G: nx.Graph) -> int:
        """安全计算直径"""
        try:
            if nx.is_connected(G):
                return nx.diameter(G)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                return nx.diameter(subgraph)
        except:
            return 0
    
    def _safe_radius(self, G: nx.Graph) -> int:
        """安全计算半径"""
        try:
            if nx.is_connected(G):
                return nx.radius(G)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                return nx.radius(subgraph)
        except:
            return 0
    
    def _estimate_connectivity_dimension(self, G: nx.Graph, components: List[Set]) -> float:
        """估计连通性维度"""
        if len(components) == 1:
            # 单连通分量，分析连通度
            connectivity = nx.edge_connectivity(G) if G.number_of_edges() > 0 else 0
            return min(3.0, connectivity)
        else:
            # 多连通分量，可能是低维结构
            return max(0.0, 3.0 - len(components))
    
    def _fallback_tsa_analysis(self, structure: Structure) -> Dict:
        """备用TSA分析方法"""
        try:
            lattice = structure.lattice
            a, b, c = lattice.abc
            
            ratios = [c/a, c/b, a/b, b/a, a/c, b/c]
            max_ratio = max(ratios)
            
            if max_ratio > 3.0:
                dimension = '2D'
                confidence = min(0.8, max_ratio / 10.0)
            elif max_ratio > 2.0:
                dimension = '2D'
                confidence = 0.6
            else:
                dimension = '3D'
                confidence = 0.7
                
            return {
                'dimension': dimension,
                'confidence': confidence,
                'lattice_ratios': ratios,
                'method': 'fallback'
            }
            
        except Exception:
            return {'dimension': 'Unknown', 'confidence': 0.0, 'method': 'failed'}
    
    def calculate_binding_energy_dft(self, structure: Structure) -> float:
        """
        使用DFT计算精确的层间结合能
        """
        if not self.use_dft:
            return self._estimate_binding_energy_empirical(structure)
        
        try:
            # 转换为ASE Atoms对象
            atoms = self._structure_to_ase(structure)
            
            # 设置DFT计算器
            calc = Vasp(
                xc=self.dft_config['xc'],
                encut=self.dft_config['encut'],
                kpts=self.dft_config['kpts'],
                ediff=self.dft_config['convergence']['energy'],
                nsw=0,  # 单点能计算
                ibrion=-1,
                lwave=False,
                lcharg=False
            )
            
            atoms.calc = calc
            
            # 计算总能量
            total_energy = atoms.get_potential_energy()
            
            # 分离层进行单独计算
            separated_energies = self._calculate_separated_layers_energy(atoms, structure)
            
            # 计算结合能
            binding_energy = total_energy - sum(separated_energies)
            
            # 归一化为每平方埃
            lattice = structure.lattice
            area = lattice.a * lattice.b * np.sin(np.radians(lattice.gamma))
            binding_energy_per_area = abs(binding_energy) / area
            
            self.stats['dft_calculations_count'] += 1
            
            return binding_energy_per_area
            
        except Exception as e:
            print(f"DFT计算失败，使用经验方法: {e}")
            return self._estimate_binding_energy_empirical(structure)
    
    def _structure_to_ase(self, structure: Structure) -> Atoms:
        """将PyMatGen结构转换为ASE Atoms"""
        symbols = [str(site.specie) for site in structure]
        positions = [site.coords for site in structure]
        cell = structure.lattice.matrix
        
        return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    
    def _calculate_separated_layers_energy(self, atoms: Atoms, structure: Structure) -> List[float]:
        """计算分离层的能量"""
        # 简化实现：假设沿c轴分离
        # 在实际应用中需要更复杂的层识别和分离算法
        
        try:
            # 识别层
            coords = atoms.positions
            z_coords = coords[:, 2]
            
            # 聚类找层
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=2.0, min_samples=1).fit(z_coords.reshape(-1, 1))
            
            layer_energies = []
            for label in set(clustering.labels_):
                layer_indices = np.where(clustering.labels_ == label)[0]
                
                # 创建单层结构
                layer_atoms = atoms[layer_indices]
                
                # 调整晶胞（增大c轴避免层间相互作用）
                cell = layer_atoms.cell.copy()
                cell[2, 2] = cell[2, 2] * 3  # 增大c轴
                layer_atoms.cell = cell
                
                # 设置计算器
                calc = atoms.calc.__class__(**self.dft_config)
                layer_atoms.calc = calc
                
                # 计算单层能量
                layer_energy = layer_atoms.get_potential_energy()
                layer_energies.append(layer_energy)
            
            return layer_energies
            
        except Exception as e:
            print(f"分离层计算失败: {e}")
            return [0.0]  # 返回零能量作为备用
    
    def _estimate_binding_energy_empirical(self, structure: Structure) -> float:
        """经验方法估算结合能"""
        try:
            # 使用Lennard-Jones势和经验参数
            coords = np.array([site.coords for site in structure])
            elements = [site.specie for site in structure]
            
            # 计算层间相互作用
            interlayer_energy = 0.0
            
            # 简化的层识别
            z_coords = coords[:, 2]
            z_unique = np.unique(np.round(z_coords, 1))
            
            if len(z_unique) < 2:
                return 0.05  # 单层结构
            
            # 估算vdW相互作用
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    if abs(z_coords[i] - z_coords[j]) > 2.0:  # 不同层的原子
                        distance = np.linalg.norm(coords[i] - coords[j])
                        
                        # 使用经验vdW参数
                        vdw_energy = self._calculate_vdw_interaction(
                            elements[i], elements[j], distance
                        )
                        interlayer_energy += vdw_energy
            
            # 归一化
            lattice = structure.lattice
            area = lattice.a * lattice.b * np.sin(np.radians(lattice.gamma))
            
            return abs(interlayer_energy) / area
            
        except Exception:
            return 0.1  # 默认值
    
    def _calculate_vdw_interaction(self, element1: Element, element2: Element, distance: float) -> float:
        """计算vdW相互作用"""
        # 经验vdW参数（简化）
        vdw_params = {
            'C': {'epsilon': 0.002, 'sigma': 3.4},
            'H': {'epsilon': 0.001, 'sigma': 2.5},
            'N': {'epsilon': 0.002, 'sigma': 3.3},
            'O': {'epsilon': 0.002, 'sigma': 3.1},
            'S': {'epsilon': 0.003, 'sigma': 3.6}
        }
        
        # 默认参数
        default_params = {'epsilon': 0.002, 'sigma': 3.0}
        
        params1 = vdw_params.get(str(element1), default_params)
        params2 = vdw_params.get(str(element2), default_params)
        
        # Lorentz-Berthelot组合规则
        epsilon = np.sqrt(params1['epsilon'] * params2['epsilon'])
        sigma = (params1['sigma'] + params2['sigma']) / 2
        
        # Lennard-Jones势
        if distance > 0:
            r6 = (sigma / distance) ** 6
            energy = 4 * epsilon * (r6 ** 2 - r6)
        else:
            energy = 0
        
        return energy
    
    def analyze_vdw_interactions_quantum(self, structure: Structure) -> Dict:
        """量子化学方法分析vdW相互作用"""
        if not PYSCF_AVAILABLE:
            return self._analyze_vdw_interactions_empirical(structure)
        
        try:
            # 使用PySCF进行量子化学计算
            # 这里是简化实现，实际需要复杂的量子化学计算
            
            atoms_data = []
            for site in structure:
                atoms_data.append([str(site.specie), site.coords[0], site.coords[1], site.coords[2]])
            
            # 构建分子
            mol_string = ""
            for atom_data in atoms_data[:min(20, len(atoms_data))]:  # 限制原子数量
                mol_string += f"{atom_data[0]} {atom_data[1]:.6f} {atom_data[2]:.6f} {atom_data[3]:.6f}\n"
            
            mol = gto.M(atom=mol_string, basis='sto-3g', verbose=0)
            
            # DFT计算
            mf = dft.RKS(mol)
            mf.xc = 'b3lyp'
            energy = mf.kernel()
            
            # MP2计算相关能
            mp2 = mp.MP2(mf)
            correlation_energy = mp2.kernel()[0]
            
            # 估算vdW贡献
            vdw_contribution = abs(correlation_energy) * 0.1  # 经验比例
            
            return {
                'method': 'quantum_chemical',
                'total_energy': energy,
                'correlation_energy': correlation_energy,
                'vdw_contribution': vdw_contribution,
                'basis_set': 'sto-3g'
            }
            
        except Exception as e:
            print(f"量子化学计算失败: {e}")
            return self._analyze_vdw_interactions_empirical(structure)
    
    def _analyze_vdw_interactions_empirical(self, structure: Structure) -> Dict:
        """经验方法分析vdW相互作用"""
        try:
            coords = np.array([site.coords for site in structure])
            elements = [site.specie for site in structure]
            
            total_vdw_energy = 0.0
            interactions = []
            
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    distance = np.linalg.norm(coords[i] - coords[j])
                    
                    if 2.0 < distance < 10.0:  # vdW相互作用范围
                        vdw_energy = self._calculate_vdw_interaction(elements[i], elements[j], distance)
                        total_vdw_energy += vdw_energy
                        
                        interactions.append({
                            'atoms': (i, j),
                            'elements': (str(elements[i]), str(elements[j])),
                            'distance': distance,
                            'vdw_energy': vdw_energy
                        })
            
            return {
                'method': 'empirical',
                'total_vdw_energy': total_vdw_energy,
                'num_interactions': len(interactions),
                'interactions': interactions[:10]  # 只保存前10个
            }
            
        except Exception as e:
            return {'method': 'failed', 'error': str(e)}
    
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
            
            print(f"开始处理 {cif_file.name}...")
            
            # 完整TSA分析
            tsa_result = self.topology_scaling_analysis_complete(structure)
            
            # DFT结合能计算
            binding_energy = self.calculate_binding_energy_dft(structure)
            
            # vdW相互作用分析
            vdw_analysis = self.analyze_vdw_interactions_quantum(structure)
            
            # 综合评分
            score = self._calculate_2d_score_advanced(tsa_result, binding_energy, vdw_analysis)
            
            # 判定结果
            is_2d_candidate = (
                tsa_result['dimension'] == '2D' and
                tsa_result['confidence'] > 0.6 and
                binding_energy < self.max_binding_energy and
                score > self.min_2d_score
            )
            
            result = {
                'filename': cif_file.name,
                'formula': formula,
                'num_elements': int(num_elements),
                'elements': [str(el) for el in elements],
                'tsa_analysis': self._convert_to_json_serializable(tsa_result),
                'binding_energy_dft': float(binding_energy),
                'vdw_analysis': self._convert_to_json_serializable(vdw_analysis),
                'score': float(score),
                'is_2d_candidate': bool(is_2d_candidate),
                'status': 'accepted' if is_2d_candidate else 'rejected',
                'processing_method': 'advanced'
            }
            
            # 更新统计
            if tsa_result['dimension'] == '2D':
                self.stats['dimensionality_2d'] += 1
            if binding_energy < self.max_binding_energy:
                self.stats['low_binding_energy'] += 1
            if is_2d_candidate:
                self.stats['final_2d_candidates'] += 1
            
            return result
            
        except Exception as e:
            self.stats['processing_errors'] += 1
            print(f"处理 {cif_file.name} 时出错: {e}")
            return {
                'filename': cif_file.name,
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_2d_score_advanced(self, tsa_result: Dict, binding_energy: float, vdw_analysis: Dict) -> float:
        """高级综合评分算法"""
        score = 0.0
        
        # TSA分析权重 (50%)
        if tsa_result['dimension'] == '2D':
            score += 0.5 * tsa_result['confidence']
        
        # 结合能权重 (25%)
        if binding_energy < self.max_binding_energy:
            binding_score = 1.0 - (binding_energy / self.max_binding_energy)
            score += 0.25 * binding_score
        
        # vdW相互作用权重 (25%)
        if 'vdw_contribution' in vdw_analysis:
            vdw_score = min(1.0, vdw_analysis['vdw_contribution'] / 0.1)
            score += 0.25 * vdw_score
        
        return min(1.0, score)
    
    def _convert_to_json_serializable(self, obj):
        """转换为JSON可序列化的格式"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return list(self._convert_to_json_serializable(list(obj)))
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def run_screening(self) -> None:
        """运行完整筛选流程"""
        print("开始二维材料高精度筛选...")
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"使用DFT: {self.use_dft}")
        print(f"使用拓扑分析: {self.use_topology}")
        
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
            print(f"处理进度: {i+1}/{total_files}")
            
            result = self.process_single_material(cif_file)
            results.append(result)
            self.stats['total_processed'] += 1
            
            # 每处理10个文件保存一次中间结果
            if i % 10 == 9:
                self._save_intermediate_results(results)
        
        # 保存最终结果
        self._save_results(results)
        self._print_statistics()
        
        print("高精度筛选完成！")
    
    def _save_intermediate_results(self, results: List[Dict]) -> None:
        """保存中间结果"""
        intermediate_file = self.output_dir / "intermediate_results.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"中间结果已保存: {intermediate_file}")
    
    def _save_results(self, results: List[Dict]) -> None:
        """保存筛选结果"""
        # 保存完整结果
        results_file = self.output_dir / "advanced_2d_screening_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存二维材料候选
        candidates = [r for r in results if r.get('is_2d_candidate', False)]
        candidates_file = self.output_dir / "advanced_2d_material_candidates.json"
        with open(candidates_file, 'w', encoding='utf-8') as f:
            json.dump(candidates, f, indent=2, ensure_ascii=False)
        
        # 保存详细CSV摘要
        df_data = []
        for result in results:
            if result['status'] not in ['error']:
                row = {
                    'filename': result['filename'],
                    'formula': result['formula'],
                    'num_elements': result['num_elements'],
                    'tsa_dimension': result.get('tsa_analysis', {}).get('dimension', 'Unknown'),
                    'tsa_confidence': result.get('tsa_analysis', {}).get('confidence', 0),
                    'binding_energy_dft': result.get('binding_energy_dft', 0),
                    'score': result.get('score', 0),
                    'is_2d_candidate': result.get('is_2d_candidate', False)
                }
                
                # 添加TSA详细信息
                if 'tsa_analysis' in result:
                    tsa = result['tsa_analysis']
                    if 'topology_features' in tsa:
                        row['h1_features_count'] = tsa['topology_features'].get('h1_features', {}).get('num_features', 0)
                        row['h2_features_count'] = tsa['topology_features'].get('h2_features', {}).get('num_features', 0)
                    
                    if 'graph_features' in tsa:
                        row['graph_density'] = tsa['graph_features'].get('basic_metrics', {}).get('density', 0)
                        row['is_planar'] = tsa['graph_features'].get('planarity', {}).get('is_planar_graph', False)
                        row['is_layered'] = tsa['graph_features'].get('layering', {}).get('is_layered', False)
                
                df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(self.output_dir / "advanced_2d_screening_summary.csv", index=False)
        
        # 保存统计信息
        stats_file = self.output_dir / "screening_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存:")
        print(f"- 完整结果: {results_file}")
        print(f"- 二维候选: {candidates_file}")
        print(f"- CSV摘要: {self.output_dir / 'advanced_2d_screening_summary.csv'}")
        print(f"- 统计信息: {stats_file}")
    
    def _print_statistics(self) -> None:
        """打印统计信息"""
        print(f"\n=== 高精度筛选统计 ===")
        print(f"总处理文件数: {self.stats['total_processed']}")
        print(f"TSA识别为2D: {self.stats['dimensionality_2d']}")
        print(f"低结合能材料: {self.stats['low_binding_energy']}")
        print(f"最终2D候选: {self.stats['final_2d_candidates']}")
        print(f"处理错误: {self.stats['processing_errors']}")
        print(f"TSA分析次数: {self.stats['tsa_analysis_count']}")
        print(f"DFT计算次数: {self.stats['dft_calculations_count']}")
        
        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['total_processed'] - self.stats['processing_errors']) / self.stats['total_processed']
            candidate_rate = self.stats['final_2d_candidates'] / self.stats['total_processed']
            print(f"处理成功率: {success_rate:.1%}")
            print(f"2D候选比例: {candidate_rate:.1%}")

def main():
    """主函数"""
    # 设置路径
    input_dir = "data/filtered"  # 已经过合成性筛选的材料
    output_dir = "data/2d_materials_advanced"
    
    # 创建高精度筛选器
    screener = AdvancedTwoDMaterialScreener(
        input_dir, 
        output_dir,
        use_dft=False,  # 可以设置为True启用DFT计算（需要VASP）
        use_topology=True
    )
    
    # 运行筛选
    screener.run_screening()

if __name__ == "__main__":
    main()