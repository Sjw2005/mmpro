"""
一键测试 results/ 下所有模型在测试集上的表现
用法: python test_all.py
"""
from ultralytics import YOLO
import warnings
import os
import gc
import torch

warnings.filterwarnings('ignore')

# 要跳过的目录（如已有的test结果目录，或名称含-test的）
SKIP_DIRS = {'DMFNet_MDAFP3D-test'}

def clean_gpu():
    """清理GPU显存，防止连续测试OOM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

if __name__ == '__main__':
    results_dir = './results'

    # 自动发现所有含 best.pt 的模型
    models = []
    for name in sorted(os.listdir(results_dir)):
        if name in SKIP_DIRS:
            continue
        best_pt = os.path.join(results_dir, name, 'weights', 'best.pt')
        if os.path.isfile(best_pt):
            models.append((name, best_pt))

    print(f"发现 {len(models)} 个模型待测试:")
    for name, path in models:
        print(f"  - {name}")
    print("=" * 60)

    summary = []

    for name, best_pt in models:
        print(f"\n{'='*60}")
        print(f"正在测试: {name}")
        print(f"{'='*60}")

        try:
            clean_gpu()
            model = YOLO(best_pt)
            metrics = model.val(
                data='data.yaml',
                imgsz=640,
                batch=16,
                split='test',
                workers=8,
                device='0',
                project='results',
                name=f'{name}-test',
                exist_ok=True,
            )

            # 收集指标
            summary.append({
                'name': name,
                'mAP50': metrics.box.map50,
                'mAP50-95': metrics.box.map,
                'precision': metrics.box.mp,
                'recall': metrics.box.mr,
            })
            del model, metrics
        except Exception as e:
            print(f"[ERROR] {name} 测试失败: {e}")
            summary.append({
                'name': name,
                'mAP50': None,
                'mAP50-95': None,
                'precision': None,
                'recall': None,
            })
        finally:
            clean_gpu()

    # 打印汇总表格
    print(f"\n\n{'='*80}")
    print("测试集结果汇总")
    print(f"{'='*80}")
    print(f"{'模型名称':<30} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print("-" * 80)

    for r in summary:
        if r['mAP50'] is not None:
            print(f"{r['name']:<30} {r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f} {r['precision']:>10.4f} {r['recall']:>8.4f}")
        else:
            print(f"{r['name']:<30} {'FAILED':>8} {'FAILED':>10} {'FAILED':>10} {'FAILED':>8}")

    print(f"{'='*80}")
