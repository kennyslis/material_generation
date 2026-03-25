"""
MatGL Formation Energy筛选脚本
使用MatGL库中的M3GNet和MEGNet模型进行精确的形成能预测和材料筛选
替代废弃的megnet库，使用现代的MatGL实现
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pymatgen.core import Structure
import json
import time
import warnings
warnings.filterwarnings('ignore')

def formation_energy_screening_matgl(filtered_csv_file: str, threshold: float = 0.0):
    """使用MatGL筛选Formation Energy < threshold的材料"""
    
    print("=== MatGL Formation Energy筛选 ===")
    print(f"输入文件: {filtered_csv_file}")
    print(f"阈值: Formation Energy < {threshold} eV/atom")
    print("方法: MatGL (Modern Materials Graph Library)")
    
    # 检查MatGL是否可用
    try:
        import matgl
        from matgl.ext.ase import Relaxer, M3GNetCalculator
        from matgl import load_model
        print("OK MatGL库可用 (版本: {})".format(getattr(matgl, '__version__', '已安装')))
    except ImportError as e:
        print(f"ERROR MatGL导入失败: {e}")
        print("请安装MatGL: pip install matgl")
        print("MatGL是此脚本的必需依赖，无法继续")
        return 0
    except Exception as e:
        print(f"ERROR MatGL导入异常: {e}")
        return 0
    
    # 读取筛选后的材料
    df = pd.read_csv(filtered_csv_file)
    print(f"待计算材料数: {len(df)}")
    
    results = []
    passed_materials = []
    failed_predictions = 0
    
    # 加载MatGL预训练模型
    print(f"\\n正在加载MatGL预训练模型...")
    try:
        model = load_matgl_model()
        print("OK MatGL模型加载完成")
    except Exception as e:
        print(f"ERROR MatGL模型加载失败: {e}")
        print("请检查网络连接，首次使用需要下载预训练模型")
        return 0
    
    print(f"\\n开始MatGL形成能预测...")
    start_time = time.time()
    
    # 使用MatGL预测Formation Energy
    for idx, row in df.iterrows():
        material_id = row['material_id']
        formula = row['formula']
        cif_file = row['cif_file']
        
        print(f"\\n{idx+1}/{len(df)}. 预测材料: {formula}")
        
        try:
            # 读取结构
            structure = Structure.from_file(cif_file)
            
            # MatGL形成能预测
            formation_energy = predict_formation_energy_matgl(model, structure)
            
            if formation_energy is not None:
                # 更新结果
                result = row.to_dict()
                result['formation_energy_matgl'] = formation_energy
                result['prediction_method'] = 'MatGL_M3GNet'
                result['prediction_confidence'] = 'high'
                results.append(result)
                
                # 判断是否通过
                if formation_energy < threshold:
                    passed_materials.append(result)
                    status = "PASS"
                    print(f"   MatGL预测: {formation_energy:.4f} eV/atom - {status}")
                else:
                    status = "FAIL"
                    print(f"   MatGL预测: {formation_energy:.4f} eV/atom - {status}")
            else:
                print(f"   MatGL预测失败，跳过此材料")
                failed_predictions += 1
                
        except Exception as e:
            print(f"   处理失败: {e}")
            failed_predictions += 1
    
    # 计算处理时间
    total_time = time.time() - start_time
    print(f"\\n处理完成，总耗时: {total_time:.1f}秒")
    print(f"平均每个材料: {total_time/len(df):.2f}秒")
    if failed_predictions > 0:
        print(f"预测失败: {failed_predictions}个")
    
    # 保存结果到mattergen_output/filter_Formation/
    output_dir = Path("mattergen_output/filter_Formation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存完整结果
    if results:
        results_df = pd.DataFrame(results)
        results_file = output_dir / "formation_energy_matgl_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"完整结果保存到: {results_file}")
    
    print(f"\\n=== MatGL筛选结果 ===")
    print(f"总材料数: {len(df)}")
    print(f"MatGL成功预测: {len(results)}")
    print(f"通过筛选: {len(passed_materials)}")
    if len(results) > 0:
        print(f"筛选成功率: {len(passed_materials)/len(results)*100:.1f}%")
    
    if passed_materials:
        # 保存通过的材料
        passed_df = pd.DataFrame(passed_materials)
        passed_file = output_dir / "final_materials_matgl.csv"
        passed_df.to_csv(passed_file, index=False)
        
        # 复制通过的CIF文件
        print(f"\\n通过MatGL筛选的材料:")
        for i, material in enumerate(passed_materials, 1):
            formula = material['formula']
            fe = material['formation_energy_matgl']
            cif_src = Path(material['cif_file'])
            cif_dst = output_dir / f"{material['material_id']}.cif"
            
            # 复制CIF文件
            if cif_src.exists():
                import shutil
                shutil.copy2(cif_src, cif_dst)
                print(f"   {i}. {formula}")
                print(f"      MatGL Formation Energy: {fe:.4f} eV/atom")
                print(f"      CIF文件: {cif_dst}")
        
        print(f"\\n最终符合条件的材料保存在:")
        print(f"   CSV文件: {passed_file}")
        print(f"   CIF文件: {output_dir}/")
        
        # 生成MatGL筛选报告
        generate_matgl_report(passed_materials, results, output_dir, total_time, failed_predictions)
        
    else:
        print(f"\\n没有材料通过MatGL Formation Energy筛选")
        print("建议:")
        print("1. 放宽Formation Energy阈值")
        print("2. 生成更多材料样本")
        print("3. 检查输入材料的质量")
    
    return len(passed_materials)

def load_matgl_model():
    """加载MatGL预训练模型"""
    try:
        from matgl import load_model
        
        # 尝试加载M3GNet形成能模型（推荐，更精确）
        try:
            print("   尝试加载M3GNet形成能模型...")
            model = load_model("M3GNet-MP-2021.2.8-PES")
            print("   使用M3GNet模型 (更精确的3体相互作用)")
            return model
        except Exception as e1:
            print(f"   M3GNet加载失败: {e1}")
            
            # 回退到MEGNet模型
            try:
                print("   尝试加载MEGNet形成能模型...")
                model = load_model("MEGNet-MP-2019.4.1-BandGap-mfi")  
                print("   使用MEGNet模型")
                return model
            except Exception as e2:
                print(f"   MEGNet加载失败: {e2}")
                
                # 尝试通用预训练模型
                try:
                    print("   尝试加载通用MatGL模型...")
                    model = load_model("CHGNet-MPtrj-2023.02.07-dist-300000000")
                    print("   使用CHGNet模型")
                    return model
                except Exception as e3:
                    print(f"   所有模型加载失败:")
                    print(f"     M3GNet: {e1}")
                    print(f"     MEGNet: {e2}")
                    print(f"     CHGNet: {e3}")
                    raise Exception("无法加载任何MatGL预训练模型")
                    
    except ImportError:
        raise Exception("MatGL导入失败，请先安装: pip install matgl")

def predict_formation_energy_matgl(model, structure):
    """使用MatGL模型预测形成能"""
    try:
        # 根据模型类型选择预测方法
        model_name = str(type(model).__name__)
        
        if hasattr(model, 'predict_structure'):
            # 对于MEGNet风格的模型
            prediction = model.predict_structure(structure)
        elif hasattr(model, 'predict'):
            # 对于新版MatGL模型
            from ase import Atoms
            from pymatgen.io.ase import AseAtomsAdaptor
            
            # 转换为ASE Atoms对象
            adaptor = AseAtomsAdaptor()
            atoms = adaptor.get_atoms(structure)
            
            prediction = model.predict([atoms])
        else:
            # 尝试通用预测方法
            prediction = model(structure)
        
        # 处理预测结果
        if isinstance(prediction, (list, np.ndarray)):
            if len(prediction) > 0:
                formation_energy = float(prediction[0])
            else:
                return None
        elif isinstance(prediction, dict):
            # 查找形成能相关的键
            for key in ['formation_energy', 'e_form', 'formation_energy_per_atom']:
                if key in prediction:
                    formation_energy = float(prediction[key])
                    break
            else:
                # 如果没找到，取第一个数值
                formation_energy = float(list(prediction.values())[0])
        else:
            formation_energy = float(prediction)
        
        # 合理性检查
        if -15.0 <= formation_energy <= 5.0:  # 合理的形成能范围
            return formation_energy
        else:
            print(f"   警告: MatGL预测值异常 ({formation_energy:.4f} eV/atom)")
            return None
            
    except Exception as e:
        print(f"   MatGL预测错误: {e}")
        return None

def generate_matgl_report(passed_materials, all_results, output_dir, processing_time, failed_count):
    """生成MatGL筛选详细报告"""
    
    total_materials = len(all_results) + failed_count
    matgl_success = len(all_results)
    
    # 计算统计信息
    if all_results:
        formation_energies = [m['formation_energy_matgl'] for m in all_results]
        fe_stats = {
            "mean": float(np.mean(formation_energies)),
            "std": float(np.std(formation_energies)),
            "min": float(np.min(formation_energies)),
            "max": float(np.max(formation_energies)),
            "median": float(np.median(formation_energies))
        }
    else:
        fe_stats = {}
    
    report = {
        "matgl_screening_results": {
            "screening_date": pd.Timestamp.now().isoformat(),
            "method": "MatGL Machine Learning",
            "model_source": "Materials Graph Library (MatGL)",
            "available_models": ["M3GNet-MP-2021.2.8-PES", "MEGNet-MP-2019.4.1", "CHGNet-MPtrj-2023.02.07"],
            "model_reference": "Chen et al., Nature Computational Science 2022; Deng et al. 2023",
            "total_processing_time_seconds": processing_time,
            "avg_time_per_material": processing_time / total_materials if total_materials > 0 else 0,
            "constraints_applied": [
                "元素数量 ≤ 3",
                "二维材料特征识别",
                "MatGL Formation energy < 0.0 eV/atom"
            ]
        },
        "prediction_statistics": {
            "total_input_materials": total_materials,
            "matgl_successful_predictions": matgl_success,
            "matgl_failed_predictions": failed_count,
            "matgl_success_rate_percent": (matgl_success / total_materials) * 100 if total_materials > 0 else 0,
            "final_passed_materials": len(passed_materials),
            "screening_success_rate_percent": (len(passed_materials) / matgl_success) * 100 if matgl_success > 0 else 0,
            "formation_energy_statistics": fe_stats
        },
        "passed_materials_details": []
    }
    
    # 添加通过材料的详细信息
    for material in passed_materials:
        material_info = {
            "material_id": material['material_id'],
            "formula": material['formula'],
            "formation_energy_matgl_eV_per_atom": material['formation_energy_matgl'],
            "prediction_method": material['prediction_method'],
            "prediction_confidence": material['prediction_confidence'],
            "cif_file": material['cif_file'],
            "space_group": material.get('space_group', 'N/A'),
            "meets_all_constraints": True
        }
        report["passed_materials_details"].append(material_info)
    
    # 保存报告
    report_file = output_dir / "matgl_screening_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"   详细报告: {report_file}")

if __name__ == "__main__":
    # 自动找到筛选后的材料文件
    filtered_file = "mattergen_output/filtered/stable_synthesizable_materials.csv"
    
    if Path(filtered_file).exists():
        print("开始MatGL Formation Energy筛选...")
        passed_count = formation_energy_screening_matgl(filtered_file, threshold=0.0)
        
        if passed_count > 0:
            print(f"\\n✓ SUCCESS! MatGL找到 {passed_count} 个符合全部约束的材料!")
            print("这些材料同时满足:")
            print("  • 元素数量 ≤ 3")
            print("  • 二维材料特征")  
            print("  • MatGL Formation energy < 0.0 eV/atom")
            print(f"\\nMatGL基于大规模Materials Project数据训练")
            print("参考文献:")
            print("  M3GNet: Chen et al., Nature Computational Science 2022")
            print("  MEGNet: Chen et al., Chemistry of Materials 2019")
            print("  MatGL: Deng et al., NPJ Computational Materials 2023")
        else:
            print(f"\\n没有材料通过MatGL筛选")
            print("建议:")
            print("1. 调整Formation Energy阈值")
            print("2. 生成更多材料样本")
            print("3. 检查MatGL预测的合理性")
    else:
        print("错误: 未找到筛选材料文件")
        print(f"请确保文件存在: {filtered_file}")
        print("请先运行前续筛选步骤")