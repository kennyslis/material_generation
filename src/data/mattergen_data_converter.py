"""
MatterGen数据转换器

将现有的CIF文件和特征数据转换为MatterGen训练所需的格式
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MatterGenDataConverter:
    """MatterGen数据格式转换器"""
    
    def __init__(self, 
                 cif_dir="data/2d_materials",
                 features_dir="data/features", 
                 output_dir="data/mattergen"):
        """
        初始化转换器
        
        Args:
            cif_dir: CIF文件目录
            features_dir: 特征文件目录
            output_dir: 输出目录
        """
        self.cif_dir = Path(cif_dir)
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """加载现有数据"""
        logger.info("加载训练和测试数据...")
        
        # 加载特征和ID
        self.train_features = np.load(self.features_dir / "train_features.npy")
        self.train_ids = pd.read_csv(self.features_dir / "train_ids.csv")
        self.test_features = np.load(self.features_dir / "test_features.npy")
        self.test_ids = pd.read_csv(self.features_dir / "test_ids.csv")
        
        logger.info(f"训练集大小: {len(self.train_features)}, 测试集大小: {len(self.test_features)}")
        logger.info(f"特征维度: {self.train_features.shape[1]}")
        
    def create_mattergen_dataset(self):
        """创建MatterGen格式的数据集"""
        logger.info("创建MatterGen数据集格式...")
        
        # 创建训练和测试数据目录
        train_dir = self.output_dir / "train"
        test_dir = self.output_dir / "test"
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        # 处理训练数据
        self._process_split(self.train_ids, self.train_features, train_dir, "train")
        
        # 处理测试数据
        self._process_split(self.test_ids, self.test_features, test_dir, "test")
        
        # 创建数据集配置文件
        self._create_dataset_config()
        
    def _process_split(self, ids_df, features, output_dir, split_name):
        """处理单个数据分割"""
        logger.info(f"处理{split_name}数据...")
        
        structures = []
        properties = []
        
        for idx, row in tqdm(ids_df.iterrows(), total=len(ids_df), desc=f"处理{split_name}"):
            material_id = row['material_id']
            feature_vector = features[idx]
            
            # 查找对应的CIF文件
            cif_file = self.cif_dir / f"{material_id}.cif"
            
            if cif_file.exists():
                try:
                    # 读取结构
                    structure = Structure.from_file(cif_file)
                    
                    # 保存结构数据
                    structure_data = {
                        "material_id": material_id,
                        "lattice": structure.lattice.matrix.tolist(),
                        "species": [str(site.specie) for site in structure],
                        "coords": structure.cart_coords.tolist(),
                        "formula": structure.composition.reduced_formula,
                        "num_atoms": len(structure),
                        "volume": structure.volume,
                        "density": structure.density
                    }
                    
                    structures.append(structure_data)
                    
                    # 保存属性数据（特征向量）
                    properties.append({
                        "material_id": material_id,
                        "features": feature_vector.tolist(),
                        "feature_dim": len(feature_vector)
                    })
                    
                except Exception as e:
                    logger.warning(f"处理{material_id}时出错: {e}")
                    continue
            else:
                logger.warning(f"找不到CIF文件: {cif_file}")
        
        # 保存结构和属性数据
        with open(output_dir / "structures.json", "w") as f:
            json.dump(structures, f, indent=2)
            
        with open(output_dir / "properties.json", "w") as f:
            json.dump(properties, f, indent=2)
            
        logger.info(f"{split_name}数据处理完成: {len(structures)}个结构")
        
    def _create_dataset_config(self):
        """创建数据集配置文件"""
        config = {
            "dataset_name": "2d_materials",
            "description": "二维材料数据集，用于扩散模型训练",
            "train_size": len(self.train_features),
            "test_size": len(self.test_features),
            "feature_dim": self.train_features.shape[1],
            "data_format": "json",
            "structure_file": "structures.json",
            "properties_file": "properties.json",
            "target_properties": ["features"],
            "created_from": {
                "cif_directory": str(self.cif_dir),
                "features_directory": str(self.features_dir)
            },
            "preprocessing": {
                "feature_extractor": "SE3FeatureExtractor",
                "normalization": "none"
            }
        }
        
        with open(self.output_dir / "dataset_config.json", "w") as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"数据集配置保存至: {self.output_dir / 'dataset_config.json'}")
        
    def create_training_csv(self):
        """创建MatterGen训练所需的CSV文件格式"""
        logger.info("创建训练CSV文件...")
        
        # 合并训练和测试数据
        all_ids = pd.concat([self.train_ids, self.test_ids], ignore_index=True)
        all_features = np.vstack([self.train_features, self.test_features])
        
        # 创建材料数据列表
        materials_data = []
        
        for idx, row in tqdm(all_ids.iterrows(), total=len(all_ids), desc="创建CSV数据"):
            material_id = row['material_id']
            feature_vector = all_features[idx]
            
            cif_file = self.cif_dir / f"{material_id}.cif"
            
            if cif_file.exists():
                try:
                    structure = Structure.from_file(cif_file)
                    
                    # 添加到材料列表
                    materials_data.append({
                        "material_id": material_id,
                        "formula": structure.composition.reduced_formula,
                        "spacegroup": structure.get_space_group_info()[1],
                        "num_atoms": len(structure),
                        "volume": structure.volume,
                        "density": structure.density,
                        "cif_file": str(cif_file),
                        "is_2d": True,  # 标记为二维材料
                        "split": "train" if idx < len(self.train_ids) else "test"
                    })
                    
                except Exception as e:
                    logger.warning(f"处理{material_id}时出错: {e}")
                    continue
        
        # 创建DataFrame并保存
        df = pd.DataFrame(materials_data)
        df.to_csv(self.output_dir / "materials_dataset.csv", index=False)
        
        logger.info(f"CSV文件保存至: {self.output_dir / 'materials_dataset.csv'}")
        logger.info(f"总材料数: {len(df)}")
        
    def convert_all(self):
        """执行完整的数据转换流程"""
        logger.info("开始MatterGen数据转换...")
        
        # 加载数据
        self.load_data()
        
        # 创建MatterGen数据集
        self.create_mattergen_dataset()
        
        # 创建训练CSV
        self.create_training_csv()
        
        logger.info("数据转换完成!")
        logger.info(f"输出目录: {self.output_dir}")
        
    def verify_conversion(self):
        """验证转换结果"""
        logger.info("验证数据转换结果...")
        
        # 检查文件是否存在
        required_files = [
            "dataset_config.json",
            "materials_dataset.csv",
            "train/structures.json",
            "train/properties.json",
            "test/structures.json", 
            "test/properties.json"
        ]
        
        for file_path in required_files:
            full_path = self.output_dir / file_path
            if full_path.exists():
                logger.info(f"✓ {file_path} 存在")
            else:
                logger.error(f"✗ {file_path} 不存在")
                
        # 检查数据一致性
        if (self.output_dir / "materials_dataset.csv").exists():
            df = pd.read_csv(self.output_dir / "materials_dataset.csv")
            logger.info(f"CSV文件包含 {len(df)} 个材料")
            logger.info(f"训练集: {len(df[df['split'] == 'train'])} 个")
            logger.info(f"测试集: {len(df[df['split'] == 'test'])} 个")


if __name__ == "__main__":
    # 创建转换器并执行转换
    converter = MatterGenDataConverter()
    
    # 执行完整转换
    converter.convert_all()
    
    # 验证结果
    converter.verify_conversion() 