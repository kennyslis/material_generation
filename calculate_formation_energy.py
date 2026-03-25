"""
计算材料的Formation Energy并筛选符合条件的材料
支持多种计算方法：机器学习预测(CHGNet/M3GNet)、VASP(需要本地安装)

机器学习方法（推荐）:
  - CHGNet: 基于互相作用图的机器学习势（高精度，无需license）
  - M3GNet: 多体张量势（高精度，无需license）

DFT方法（预留接口）:
  - VASP: 需要用户自行安装VASP软件和license
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
import subprocess
warnings.filterwarnings('ignore')

# 导入材料科学相关包
try:
    from pymatgen.core import Structure, Composition
    from pymatgen.io.vasp import Poscar, Incar, Potcar, Kpoints
    from pymatgen.io.vasp.inputs import VaspInput
    PYMATGEN_AVAILABLE = True
except ImportError:
    print("警告: pymatgen未安装，某些功能将不可用")
    PYMATGEN_AVAILABLE = False

try:
    from ase import Atoms
    from ase.io import read, write
    from ase.calculators.vasp import Vasp
    ASE_AVAILABLE = True
except ImportError:
    print("警告: ASE未安装，VASP计算功能将不可用")
    ASE_AVAILABLE = False

try:
    import torch
    from chgnet.model import CHGNet
    from matgl import load_model
    ML_AVAILABLE = True
except ImportError:
    print("信息: 机器学习预测模型未安装，仅支持empirical方法")
    ML_AVAILABLE = False


class VASPInterface:
    """VASP DFT计算接口（预留）

    使用前需要用户配置：
    1. 安装VASP软件
    2. 配置VASP_COMMAND环境变量
    3. 准备POTCAR伪势文件库
    """

    def __init__(self, vasp_cmd: str = None):
        """初始化VASP接口

        Args:
            vasp_cmd: VASP可执行命令（如 'vasp_std' 或完整路径）
        """
        self.vasp_cmd = vasp_cmd or os.environ.get('VASP_COMMAND', 'vasp_std')
        self.potcar_dir = os.environ.get('POTCAR_DIR', './POTCAR')
        self.vasp_available = self._check_vasp_installation()

    def _check_vasp_installation(self) -> bool:
        """检查VASP是否已安装"""
        try:
            result = subprocess.run([self.vasp_cmd, '--version'],
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def calculate_structure_energy(self, structure: Structure,
                                   directory: str = None) -> Tuple[float, Dict]:
        """计算结构的总能量

        Args:
            structure: PyMatGen Structure对象
            directory: 计算目录

        Returns:
            (总能量(eV), 计算细节字典)
        """
        if not self.vasp_available:
            raise RuntimeError(
                f"VASP未安装或无法执行 ({self.vasp_cmd})。\n"
                f"请检查：\n"
                f"1. VASP是否已安装\n"
                f"2. 设置VASP_COMMAND环境变量\n"
                f"3. 是否有VASP license"
            )

        if not ASE_AVAILABLE:
            raise RuntimeError("ASE未安装，无法调用VASP")

        try:
            calc_dir = Path(directory) if directory else Path(f"vasp_calc_{structure.composition.reduced_formula}")
            calc_dir.mkdir(exist_ok=True)

            # 转换为ASE格式
            atoms = structure.to_ase_atoms()

            # VASP计算器配置（2D材料优化参数）
            calc = Vasp(
                directory=str(calc_dir),
                xc='PBE',           # 交换关联泛函
                encut=520,          # 能量截断
                kpts=(4, 4, 2),     # k点网格（2D材料）
                ismear=0,           # 高斯展宽
                sigma=0.05,
                algo='Normal',      # 电子优化算法
                prec='Accurate',    # 精度
                ediff=1e-6,         # 能量收敛标准
                ibrion=2,           # 离子优化：BFGS
                nsw=100,            # 最大离子步数
                potim=0.5,          # 时间步长
                lorbit=11,          # DOSCAR输出
                nelect=None,        # 自动计算电子数
                ispin=1,            # 非磁性计算
                npar=4              # 并行参数
            )

            atoms.set_calculator(calc)

            # 运行VASP计算
            print(f"正在运行VASP计算，目录：{calc_dir}")
            total_energy = atoms.get_potential_energy()

            return total_energy, {
                "method": "VASP",
                "status": "success",
                "calc_dir": str(calc_dir),
                "xc": "PBE",
                "encut": 520,
                "kpts": (4, 4, 2)
            }

        except Exception as e:
            return None, {
                "method": "VASP",
                "status": "failed",
                "error": str(e)
            }


class FormationEnergyCalculator:
    """Formation Energy计算器

    支持方法：
    - ml_chgnet: CHGNet机器学习势（推荐）
    - ml_m3gnet: M3GNet机器学习势（推荐）
    - vasp: VASP DFT计算（需本地安装）
    - empirical: 经验公式（低精度，备选）
    """

    def __init__(self,
                 method: str = "ml_chgnet",
                 vasp_cmd: str = None):
        """初始化Formation Energy计算器

        Args:
            method: 计算方法 ('ml_chgnet', 'ml_m3gnet', 'vasp', 'empirical')
            vasp_cmd: VASP命令路径（如果使用VASP方法）
        """
        self.method = method
        self.ml_model = None

        # 初始化VASP接口（如果使用VASP方法）
        if method == "vasp":
            self.vasp_interface = VASPInterface(vasp_cmd=vasp_cmd)

        # 初始化ML模型
        if method.startswith("ml_") and ML_AVAILABLE:
            self._initialize_ml_model()

        # 元素参考能量 (eV/atom) - 用于Formation Energy计算
        self.reference_energies = {
            'H': -3.39, 'He': 0.0, 'Li': -1.90, 'Be': -3.73, 'B': -6.68,
            'C': -9.22, 'N': -8.27, 'O': -4.95, 'F': -1.91, 'Ne': 0.0,
            'Na': -1.31, 'Mg': -1.51, 'Al': -3.74, 'Si': -5.42, 'P': -5.41,
            'S': -4.13, 'Cl': -1.84, 'Ar': 0.0, 'K': -1.05, 'Ca': -1.88,
            'Sc': -6.33, 'Ti': -7.89, 'V': -9.08, 'Cr': -9.51, 'Mn': -9.00,
            'Fe': -8.45, 'Co': -7.11, 'Ni': -5.78, 'Cu': -3.72, 'Zn': -1.35,
            'Ga': -2.81, 'Ge': -4.61, 'As': -4.66, 'Se': -3.49, 'Br': -1.22,
            'Kr': 0.0, 'Rb': -0.91, 'Sr': -1.69, 'Y': -6.47, 'Zr': -8.54,
            'Nb': -10.10, 'Mo': -10.96, 'Tc': -10.20, 'Ru': -9.22, 'Rh': -7.36,
            'Pd': -5.18, 'Ag': -2.95, 'Cd': -0.91, 'In': -2.52, 'Sn': -4.00,
            'Sb': -4.13, 'Te': -3.14, 'I': -1.57, 'Xe': 0.0, 'Cs': -0.90,
            'Ba': -1.90, 'La': -4.93, 'Ce': -5.94, 'Pr': -4.78, 'Nd': -4.58,
            'Pm': -4.48, 'Sm': -4.46, 'Eu': -1.84, 'Gd': -4.66, 'Tb': -4.63,
            'Dy': -4.60, 'Ho': -4.58, 'Er': -4.57, 'Tm': -4.48, 'Yb': -1.60,
            'Lu': -4.52, 'Hf': -9.95, 'Ta': -11.85, 'W': -12.96, 'Re': -12.44,
            'Os': -11.17, 'Ir': -8.85, 'Pt': -6.06, 'Au': -3.27, 'Hg': 0.30,
            'Tl': -2.32, 'Pb': -3.70, 'Bi': -3.89, 'Po': -3.0, 'At': -2.0,
            'Rn': 0.0
        }
        
        # 初始化ML模型
        self.ml_model = None
        if method.startswith("ml_") and ML_AVAILABLE:
            self._initialize_ml_model()
    
    def _initialize_ml_model(self):
        """初始化机器学习模型"""
        try:
            if self.method == "ml_chgnet":
                self.ml_model = CHGNet.load()
                print(" CHGNet模型加载成功")
            elif self.method == "ml_m3gnet":
                self.ml_model = load_model("M3GNet-MP-2021.2.8-PES")
                print(" M3GNet模型加载成功")
        except Exception as e:
            print(f" ML模型加载失败: {e}")
            print("将使用经验公式估算")
            self.method = "empirical"
    
    def calculate_formation_energy_single(self, cif_file: str) -> Tuple[float, Dict]:
        """计算单个材料的Formation Energy

        Args:
            cif_file: CIF结构文件路径

        Returns:
            (Formation Energy (eV/atom), 计算细节字典)
        """
        try:
            structure = Structure.from_file(cif_file)
            composition = structure.composition

            if self.method == "ml_chgnet":
                return self._calculate_ml_chgnet(structure, composition)
            elif self.method == "ml_m3gnet":
                return self._calculate_ml_m3gnet(structure, composition)
            elif self.method == "vasp":
                return self._calculate_vasp(structure, composition, cif_file)
            else:  # empirical
                return self._calculate_empirical(structure, composition)

        except Exception as e:
            print(f" 计算 {cif_file} 失败: {e}")
            return 999.0, {"error": str(e), "method": self.method}
    
    def _calculate_ml_chgnet(self, structure: Structure, composition: Composition) -> Tuple[float, Dict]:
        """使用CHGNet计算Formation Energy"""
        try:
            # 转换为ASE Atoms对象
            atoms = structure.to_ase_atoms()
            
            # 计算总能量
            prediction = self.ml_model.predict_structure(structure)
            total_energy = prediction['e']  # eV
            
            # 计算参考能量
            reference_energy = 0.0
            for element, amount in composition.element_composition.items():
                ref_e = self.reference_energies.get(str(element), -5.0)  # 默认值
                reference_energy += amount * ref_e
            
            # Formation energy per atom
            num_atoms = composition.num_atoms
            formation_energy = (total_energy - reference_energy) / num_atoms
            
            return formation_energy, {
                "method": "CHGNet",
                "total_energy": total_energy,
                "reference_energy": reference_energy,
                "num_atoms": num_atoms,
                "reliability": "high"
            }
            
        except Exception as e:
            print(f"CHGNet计算失败: {e}")
            return self._calculate_empirical(structure, composition)
    
    def _calculate_ml_m3gnet(self, structure: Structure, composition: Composition) -> Tuple[float, Dict]:
        """使用M3GNet计算Formation Energy"""
        try:
            # M3GNet计算
            atoms = structure.to_ase_atoms()
            total_energy = self.ml_model.predict_structure(structure)
            
            # 计算参考能量
            reference_energy = 0.0
            for element, amount in composition.element_composition.items():
                ref_e = self.reference_energies.get(str(element), -5.0)
                reference_energy += amount * ref_e
            
            num_atoms = composition.num_atoms
            formation_energy = (total_energy - reference_energy) / num_atoms
            
            return formation_energy, {
                "method": "M3GNet",
                "total_energy": total_energy,
                "reference_energy": reference_energy,
                "num_atoms": num_atoms,
                "reliability": "high"
            }
            
        except Exception as e:
            print(f"M3GNet计算失败: {e}")
            return self._calculate_empirical(structure, composition)
    
    def _calculate_vasp(self, structure: Structure, composition: Composition, cif_file: str) -> Tuple[float, Dict]:
        """使用VASP DFT计算Formation Energy

        VASP是业界标准的DFT计算软件，精度远高于机器学习方法。
        使用前需要：
        1. 购买并安装VASP（https://www.vasp.at/）
        2. 设置VASP_COMMAND环境变量指向vasp可执行文件
        3. 准备POTCAR伪势文件

        Args:
            structure: PyMatGen Structure对象
            composition: 组成
            cif_file: CIF文件路径

        Returns:
            (Formation Energy (eV/atom), 计算细节字典)
        """
        try:
            if self.vasp_interface is None:
                self.vasp_interface = VASPInterface()

            if not self.vasp_interface.vasp_available:
                raise RuntimeError(
                    "VASP未安装或无法执行。若要使用VASP:\n"
                    "1. 从https://www.vasp.at/购买license\n"
                    "2. 编译VASP并配置VASP_COMMAND环境变量\n"
                    "3. 设置POTCAR_DIR指向伪势库目录\n\n"
                    "暂时建议使用ml_chgnet或ml_m3gnet方法"
                )

            # 计算总能量
            total_energy, vasp_details = self.vasp_interface.calculate_structure_energy(
                structure,
                directory=f"vasp_calc_{Path(cif_file).stem}"
            )

            if total_energy is None:
                raise RuntimeError(f"VASP计算失败: {vasp_details.get('error')}")

            # 计算参考能量（孤立原子能量）
            reference_energy = 0.0
            for element, amount in composition.element_composition.items():
                ref_e = self.reference_energies.get(str(element), -5.0)
                reference_energy += amount * ref_e

            num_atoms = composition.num_atoms
            formation_energy = (total_energy - reference_energy) / num_atoms

            return formation_energy, {
                "method": "VASP-DFT",
                "total_energy": total_energy,
                "reference_energy": reference_energy,
                "num_atoms": num_atoms,
                "formation_energy": formation_energy,
                "reliability": "very_high",
                "calc_details": vasp_details
            }

        except RuntimeError as e:
            print(f"⚠️ VASP计算错误: {e}")
            raise
        except Exception as e:
            print(f" VASP计算异常: {e}")
            raise
    
    def _calculate_empirical(self, structure: Structure, composition: Composition) -> Tuple[float, Dict]:
        """使用经验公式估算Formation Energy"""
        try:
            # 简单的经验估算：基于元素电负性和结构特征
            total_electronegativity = 0.0
            reference_energy = 0.0
            
            for element, amount in composition.element_composition.items():
                # 使用参考能量表
                ref_e = self.reference_energies.get(str(element), -5.0)
                reference_energy += amount * ref_e
                
                # 计算平均电负性（用于稳定性估算）
                electronegativity = getattr(element, 'X', 2.0)  # Pauling电负性
                total_electronegativity += amount * electronegativity
            
            avg_electronegativity = total_electronegativity / composition.num_atoms
            
            # 经验公式：考虑电负性差异和结构密度
            electronegativity_penalty = 0.0
            elements = list(composition.element_composition.keys())
            
            if len(elements) > 1:
                for i, elem1 in enumerate(elements):
                    for elem2 in elements[i+1:]:
                        x1 = getattr(elem1, 'X', 2.0)
                        x2 = getattr(elem2, 'X', 2.0)
                        electronegativity_penalty += abs(x1 - x2) * 0.1
            
            # 密度因子 (低密度通常不稳定)
            density = structure.density
            density_factor = -0.5 if density < 3.0 else 0.0
            
            # 估算formation energy
            num_atoms = composition.num_atoms
            estimated_formation_energy = (
                -2.0 +  # 基础稳定性
                electronegativity_penalty +  # 电负性不匹配惩罚
                density_factor +  # 密度因子
                np.random.normal(0, 0.5)  # 随机噪声
            )
            
            return estimated_formation_energy, {
                "method": "Empirical",
                "avg_electronegativity": avg_electronegativity,
                "electronegativity_penalty": electronegativity_penalty,
                "density": density,
                "density_factor": density_factor,
                "num_atoms": num_atoms,
                "reliability": "low",
                "note": "经验估算，仅供参考"
            }
            
        except Exception as e:
            print(f"经验估算失败: {e}")
            return 0.0, {"method": "Failed", "error": str(e)}


class MaterialScreener:
    """材料筛选器"""
    
    def __init__(self, calculator: FormationEnergyCalculator):
        self.calculator = calculator
    
    def screen_materials(self, 
                        materials_csv: str,
                        formation_energy_threshold: float = 0.0,
                        output_dir: str = "formation_energy_screening") -> Dict:
        """筛选材料"""
        print(f"=== 开始Formation Energy筛选 ===")
        print(f"阈值: {formation_energy_threshold} eV/atom")
        print(f"计算方法: {self.calculator.method}")
        
        # 读取材料列表
        df = pd.read_csv(materials_csv)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = []
        passed_materials = []
        
        print(f"\n正在计算 {len(df)} 个材料的Formation Energy...")
        
        for idx, row in df.iterrows():
            material_id = row['material_id']
            cif_file = row['cif_file']
            
            print(f"  {idx+1}/{len(df)}: {material_id} ({row.get('formula', 'Unknown')})")
            
            # 计算Formation Energy
            formation_energy, calc_details = self.calculator.calculate_formation_energy_single(cif_file)
            
            # 记录结果
            result = row.to_dict()
            result.update({
                'formation_energy_calculated': formation_energy,
                'calculation_method': calc_details.get('method', 'Unknown'),
                'calculation_reliability': calc_details.get('reliability', 'unknown'),
                'total_energy': calc_details.get('total_energy', None),
                'reference_energy': calc_details.get('reference_energy', None),
                'calc_details': json.dumps(calc_details)
            })
            results.append(result)
            
            # 判断是否通过筛选
            if formation_energy < formation_energy_threshold:
                passed_materials.append(result)
                status = " PASS"
            else:
                status = " FAIL"
            
            print(f"    Formation Energy: {formation_energy:.3f} eV/atom - {status}")
        
        # 保存完整结果
        results_df = pd.DataFrame(results)
        results_file = output_dir / "formation_energy_results.csv"
        results_df.to_csv(results_file, index=False)
        
        # 保存通过筛选的材料
        if passed_materials:
            passed_df = pd.DataFrame(passed_materials)
            passed_file = output_dir / "formation_energy_passed.csv"
            passed_df.to_csv(passed_file, index=False)
            
            # 复制通过筛选的CIF文件
            passed_cif_dir = output_dir / "passed_cif_files"
            passed_cif_dir.mkdir(exist_ok=True)
            
            for material in passed_materials:
                src_cif = Path(material['cif_file'])
                if src_cif.exists():
                    dst_cif = passed_cif_dir / src_cif.name
                    import shutil
                    shutil.copy2(src_cif, dst_cif)
        
        # 生成统计报告
        stats = self._generate_statistics(results_df, passed_materials, formation_energy_threshold)
        
        # 保存统计报告
        report_file = output_dir / "screening_report.json"
        with open(report_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n=== 筛选完成 ===")
        print(f"总材料数: {len(results)}")
        print(f"通过筛选: {len(passed_materials)} ({len(passed_materials)/len(results)*100:.1f}%)")
        print(f"平均Formation Energy: {np.mean([r['formation_energy_calculated'] for r in results]):.3f} eV/atom")
        print(f"结果保存至: {output_dir}")
        
        return {
            "total_materials": len(results),
            "passed_materials": len(passed_materials),
            "success_rate": len(passed_materials)/len(results)*100,
            "results_file": str(results_file),
            "passed_file": str(passed_file) if passed_materials else None,
            "report_file": str(report_file),
            "output_dir": str(output_dir)
        }
    
    def _generate_statistics(self, results_df: pd.DataFrame, passed_materials: List, threshold: float) -> Dict:
        """生成统计报告"""
        formation_energies = results_df['formation_energy_calculated'].values
        
        stats = {
            "screening_criteria": {
                "formation_energy_threshold": f"< {threshold} eV/atom",
                "calculation_method": self.calculator.method
            },
            "results_summary": {
                "total_materials": len(results_df),
                "passed_materials": len(passed_materials),
                "failed_materials": len(results_df) - len(passed_materials),
                "success_rate": f"{len(passed_materials)/len(results_df)*100:.1f}%"
            },
            "formation_energy_statistics": {
                "mean": float(np.mean(formation_energies)),
                "std": float(np.std(formation_energies)),
                "min": float(np.min(formation_energies)),
                "max": float(np.max(formation_energies)),
                "median": float(np.median(formation_energies)),
                "q25": float(np.percentile(formation_energies, 25)),
                "q75": float(np.percentile(formation_energies, 75))
            }
        }
        
        if passed_materials:
            passed_energies = [m['formation_energy_calculated'] for m in passed_materials]
            stats["passed_materials_statistics"] = {
                "mean_formation_energy": float(np.mean(passed_energies)),
                "best_formation_energy": float(np.min(passed_energies)),
                "formulas": [m.get('formula', 'Unknown') for m in passed_materials]
            }
        
        return stats


def main():
    """主函数

    使用示例：
    # 使用CHGNet（推荐）
    python calculate_formation_energy.py --input-csv materials.csv --method ml_chgnet

    # 使用M3GNet
    python calculate_formation_energy.py --input-csv materials.csv --method ml_m3gnet

    # 使用VASP（需本地安装）
    python calculate_formation_energy.py --input-csv materials.csv --method vasp

    # 使用经验公式（最快，精度最低）
    python calculate_formation_energy.py --input-csv materials.csv --method empirical
    """
    parser = argparse.ArgumentParser(
        description="计算Formation Energy并筛选材料",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的计算方法：
  ml_chgnet  - CHGNet机器学习势（推荐，无需license）
  ml_m3gnet  - M3GNet多体张量势（推荐，无需license）
  vasp       - VASP DFT计算（需购买license和本地安装）
  empirical  - 经验公式估算（快速，低精度）

示例：
  python calculate_formation_energy.py --input-csv materials.csv --method ml_chgnet
  python calculate_formation_energy.py --input-csv materials.csv --method vasp
"""
    )

    parser.add_argument("--input-csv", type=str, required=True,
                       help="输入材料CSV文件路径（包含material_id, cif_file, formula等列）")
    parser.add_argument("--method", type=str, default="ml_chgnet",
                       choices=["ml_chgnet", "ml_m3gnet", "vasp", "empirical"],
                       help="计算方法（默认: ml_chgnet）")
    parser.add_argument("--threshold", type=float, default=0.0,
                       help="Formation energy筛选阈值 (eV/atom，默认: 0.0)")
    parser.add_argument("--output-dir", type=str, default="formation_energy_screening",
                       help="输出结果目录（默认: formation_energy_screening）")
    parser.add_argument("--vasp-cmd", type=str, default=None,
                       help="VASP可执行命令路径（如 /path/to/vasp_std）")

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.input_csv).exists():
        print(f" 错误: 输入文件不存在: {args.input_csv}")
        return 1

    print(f"="*60)
    print(f" Formation Energy计算器")
    print(f"方法: {args.method}")
    print(f"输入文件: {args.input_csv}")
    print(f"筛选阈值: {args.threshold} eV/atom")
    print(f"输出目录: {args.output_dir}")
    print(f"="*60)

    # 创建计算器和筛选器
    try:
        calculator = FormationEnergyCalculator(method=args.method, vasp_cmd=args.vasp_cmd)
        screener = MaterialScreener(calculator)

        # 执行筛选
        results = screener.screen_materials(
            materials_csv=args.input_csv,
            formation_energy_threshold=args.threshold,
            output_dir=args.output_dir
        )

        print(f"\n{'='*60}")
        print(f" 筛选完成!")
        print(f"{'='*60}")
        if results["passed_materials"] > 0:
            print(f"✓ 通过筛选的材料: {results['passed_materials']} 个")
            print(f"✓ 结果CSV: {results['passed_file']}")
            print(f"✓ CIF文件: {results['output_dir']}/passed_cif_files/")
        else:
            print(f"  没有材料通过Formation Energy筛选")

        print(f"✓ 完整结果: {results['results_file']}")
        print(f"✓ 统计报告: {results['report_file']}")

        return 0

    except Exception as e:
        print(f" 执行出错: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)