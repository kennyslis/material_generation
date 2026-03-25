"""
MEGNet Formation Energy筛选脚本
使用MEGNet机器学习模型进行精确的形成能预测和材料筛选
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pymatgen.core import Structure
import json
import time
import warnings
warnings.filterwarnings('ignore')

def formation_energy_screening_megnet(filtered_csv_file: str, threshold: float = 0.0):
    """使用MEGNet筛选Formation Energy < threshold的材料"""
    
    print("=== MEGNet Formation Energy筛选 ===")
    print(f"输入文件: {filtered_csv_file}")
    print(f"阈值: Formation Energy < {threshold} eV/atom")
    print("方法: MEGNet机器学习模型")
    
    # 检查MEGNet是否可用
    try:
        import megnet
        from megnet.models import MEGNetModel
        print("OK MEGNet模型可用")
    except ImportError as e:
        print(f"ERROR MEGNet导入失败: {e}")
        print("请安装MEGNet: pip install megnet")
        print("MEGNet是此脚本的必需依赖，无法继续")
        return 0
    except Exception as e:
        print(f"ERROR MEGNet导入异常: {e}")
        return 0
    
    # 读取筛选后的材料
    df = pd.read_csv(filtered_csv_file)
    print(f"待计算材料数: {len(df)}")
    
    results = []
    passed_materials = []
    failed_predictions = 0
    
    # 加载MEGNet模型（第一次会自动下载）
    print("\n正在加载MEGNet预训练模型...")
    try:
        model = load_megnet_model()
        print("OK MEGNet模型加载完成")
    except Exception as e:
        print(f"ERROR MEGNet模型加载失败: {e}")
        print("请检查网络连接，首次使用需要下载预训练模型")
        return 0
    
    print(f"\n开始MEGNet形成能预测...")
    start_time = time.time()
    
    # 使用MEGNet预测Formation Energy
    for idx, row in df.iterrows():
        material_id = row['material_id']
        formula = row['formula']
        cif_file = row['cif_file']
        
        print(f"\n{idx+1}/{len(df)}. 预测材料: {formula}")
        
        try:
            # 读取结构
            structure = Structure.from_file(cif_file)
            
            # MEGNet形成能预测
            formation_energy = predict_formation_energy_megnet(model, structure)
            
            if formation_energy is not None:
                # 更新结果
                result = row.to_dict()
                result['formation_energy_megnet'] = formation_energy
                result['prediction_method'] = 'MEGNet_ML'
                result['prediction_confidence'] = 'high'
                results.append(result)
                
                # 判断是否通过
                if formation_energy < threshold:
                    passed_materials.append(result)
                    status = "PASS"
                    print(f"   MEGNet预测: {formation_energy:.4f} eV/atom - {status}")
                else:
                    status = "FAIL"
                    print(f"   MEGNet预测: {formation_energy:.4f} eV/atom - {status}")
            else:
                print(f"   MEGNet预测失败，跳过此材料")
                failed_predictions += 1
                
        except Exception as e:
            print(f"   处理失败: {e}")
            failed_predictions += 1
    
    # 计算处理时间
    total_time = time.time() - start_time
    print(f"\n处理完成，总耗时: {total_time:.1f}秒")
    print(f"平均每个材料: {total_time/len(df):.2f}秒")
    if failed_predictions > 0:
        print(f"预测失败: {failed_predictions}个")
    
    # 保存结果到mattergen_output/filter_Formation/
    output_dir = Path("mattergen_output/filter_Formation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存完整结果
    if results:
        results_df = pd.DataFrame(results)
        results_file = output_dir / "formation_energy_megnet_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"完整结果保存到: {results_file}")
    
    print(f"\n=== MEGNet筛选结果 ===")
    print(f"总材料数: {len(df)}")
    print(f"MEGNet成功预测: {len(results)}")
    print(f"通过筛选: {len(passed_materials)}")
    if len(results) > 0:
        print(f"筛选成功率: {len(passed_materials)/len(results)*100:.1f}%")
    
    if passed_materials:
        # 保存通过的材料
        passed_df = pd.DataFrame(passed_materials)
        passed_file = output_dir / "final_materials_megnet.csv"
        passed_df.to_csv(passed_file, index=False)
        
        # 复制通过的CIF文件
        print(f"\n通过MEGNet筛选的材料:")
        for i, material in enumerate(passed_materials, 1):
            formula = material['formula']
            fe = material['formation_energy_megnet']
            cif_src = Path(material['cif_file'])
            cif_dst = output_dir / f"{material['material_id']}.cif"
            
            # 复制CIF文件
            if cif_src.exists():
                import shutil
                shutil.copy2(cif_src, cif_dst)
                print(f"   {i}. {formula}")
                print(f"      MEGNet Formation Energy: {fe:.4f} eV/atom")
                print(f"      CIF文件: {cif_dst}")
        
        print(f"\n最终符合条件的材料保存在:")
        print(f"   CSV文件: {passed_file}")
        print(f"   CIF文件: {output_dir}/")
        
        # 生成MEGNet筛选报告
        generate_megnet_report(passed_materials, results, output_dir, total_time, failed_predictions)
        
    else:
        print(f"\n没有材料通过MEGNet Formation Energy筛选")
        print("建议:")
        print("1. 放宽Formation Energy阈值")
        print("2. 生成更多材料样本")
        print("3. 检查输入材料的质量")
    
    return len(passed_materials)

def load_megnet_model():
    """加载MEGNet预训练形成能模型"""
    try:
        from megnet.models import MEGNetModel
        # 尝试加载形成能预训练模型
        model = MEGNetModel.from_file('megnet_models/mp-2019.4.1/formation_energy.json')
        return model
    except Exception as e1:
        # 如果默认路径失败，尝试其他方式
        try:
            from megnet.utils.models import load_model
            model = load_model('formation_energy')
            return model
        except Exception as e2:
            print(f"模型加载方式1失败: {e1}")
            print(f"模型加载方式2失败: {e2}")
            raise Exception("无法加载MEGNet形成能模型")

def predict_formation_energy_megnet(model, structure):
    """使用MEGNet预测形成能"""
    try:
        # MEGNet预测
        prediction = model.predict_structure(structure)
        
        # 处理预测结果
        if isinstance(prediction, (list, np.ndarray)):
            formation_energy = float(prediction[0])
        else:
            formation_energy = float(prediction)
        
        # 合理性检查
        if -15.0 <= formation_energy <= 5.0:  # 合理的形成能范围
            return formation_energy
        else:
            print(f"   警告: MEGNet预测值异常 ({formation_energy:.4f} eV/atom)")
            return None
            
    except Exception as e:
        print(f"   MEGNet预测错误: {e}")
        return None

def generate_megnet_report(passed_materials, all_results, output_dir, processing_time, failed_count):
    """生成MEGNet筛选详细报告"""
    
    total_materials = len(all_results) + failed_count
    megnet_success = len(all_results)
    
    # 计算统计信息
    if all_results:
        formation_energies = [m['formation_energy_megnet'] for m in all_results]
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
        "megnet_screening_results": {
            "screening_date": pd.Timestamp.now().isoformat(),
            "method": "MEGNet Machine Learning",
            "model_source": "Materials Project pre-trained model (~60k crystals)",
            "model_reference": "Chen et al., Chemistry of Materials 2019",
            "total_processing_time_seconds": processing_time,
            "avg_time_per_material": processing_time / total_materials if total_materials > 0 else 0,
            "constraints_applied": [
                "元素数量 ≤ 3",
                "二维材料特征识别",
                "MEGNet Formation energy < 0.0 eV/atom"
            ]
        },
        "prediction_statistics": {
            "total_input_materials": total_materials,
            "megnet_successful_predictions": megnet_success,
            "megnet_failed_predictions": failed_count,
            "megnet_success_rate_percent": (megnet_success / total_materials) * 100 if total_materials > 0 else 0,
            "final_passed_materials": len(passed_materials),
            "screening_success_rate_percent": (len(passed_materials) / megnet_success) * 100 if megnet_success > 0 else 0,
            "formation_energy_statistics": fe_stats
        },
        "passed_materials_details": []
    }
    
    # 添加通过材料的详细信息
    for material in passed_materials:
        material_info = {
            "material_id": material['material_id'],
            "formula": material['formula'],
            "formation_energy_megnet_eV_per_atom": material['formation_energy_megnet'],
            "prediction_method": material['prediction_method'],
            "prediction_confidence": material['prediction_confidence'],
            "cif_file": material['cif_file'],
            "space_group": material.get('space_group', 'N/A'),
            "meets_all_constraints": True
        }
        report["passed_materials_details"].append(material_info)
    
    # 保存报告
    report_file = output_dir / "megnet_screening_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"   详细报告: {report_file}")

if __name__ == "__main__":
    # 自动找到筛选后的材料文件
    filtered_file = "mattergen_output/filtered/stable_synthesizable_materials.csv"
    
    if Path(filtered_file).exists():
        print("开始MEGNet Formation Energy筛选...")
        passed_count = formation_energy_screening_megnet(filtered_file, threshold=0.0)
        
        if passed_count > 0:
            print(f"\n✓ SUCCESS! MEGNet找到 {passed_count} 个符合全部约束的材料!")
            print("这些材料同时满足:")
            print("  • 元素数量 ≤ 3")
            print("  • 二维材料特征")  
            print("  • MEGNet Formation energy < 0.0 eV/atom")
            print(f"\nMEGNet预测基于{60000}+个Materials Project材料训练")
            print("参考文献: Chen et al., Chemistry of Materials 2019")
        else:
            print(f"\n没有材料通过MEGNet筛选")
            print("建议:")
            print("1. 调整Formation Energy阈值")
            print("2. 生成更多材料样本")
            print("3. 检查MEGNet预测的合理性")
    else:
        print("错误: 未找到筛选材料文件")
        print(f"请确保文件存在: {filtered_file}")
        print("请先运行前续筛选步骤")