"""
批量评估脚本：对所有模型的best.pt分别跑验证集和测试集，
结果保存到各模型的实验文件夹中的 val_results.txt 和 test_results.txt
"""
import os
import sys
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# 所有模型实验文件夹（results目录下有best.pt的）
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# 自动发现所有有best.pt的模型
models = {}
for name in os.listdir(RESULTS_DIR):
    best_pt = os.path.join(RESULTS_DIR, name, 'weights', 'best.pt')
    if os.path.isfile(best_pt):
        models[name] = best_pt

print(f"发现 {len(models)} 个模型: {list(models.keys())}\n")


def format_results(metrics):
    """将验证指标格式化为可读文本"""
    lines = []

    # 整体指标
    lines.append("=" * 60)
    lines.append("整体表现")
    lines.append("=" * 60)
    lines.append(f"  mAP50:       {metrics.box.map50:.4f}")
    lines.append(f"  mAP50-95:    {metrics.box.map:.4f}")
    lines.append(f"  Precision:   {metrics.box.mp:.4f}")
    lines.append(f"  Recall:      {metrics.box.mr:.4f}")
    lines.append("")

    # 各类别指标
    lines.append("=" * 60)
    lines.append("各类别表现")
    lines.append("=" * 60)
    names = metrics.names  # {0: 'car', 1: 'bus', ...}
    lines.append(f"  {'类别':<15} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}")
    lines.append(f"  {'-'*55}")

    ap50 = metrics.box.ap50       # 每类的AP50
    ap = metrics.box.ap           # 每类的AP50-95 (如果有)
    p = metrics.box.p             # 每类precision
    r = metrics.box.r             # 每类recall

    for i, cls_name in names.items():
        if i < len(ap50):
            line = f"  {cls_name:<15} {p[i]:>10.4f} {r[i]:>10.4f} {ap50[i]:>10.4f}"
            if ap is not None and i < len(ap):
                line += f" {ap[i]:>10.4f}"
            lines.append(line)

    return "\n".join(lines)


def evaluate_model(model_name, best_pt_path):
    """对单个模型跑val和test"""
    model_dir = os.path.join(RESULTS_DIR, model_name)
    print(f"\n{'#' * 60}")
    print(f"  评估模型: {model_name}")
    print(f"  权重: {best_pt_path}")
    print(f"{'#' * 60}")

    model = YOLO(best_pt_path)

    # --- 验证集 ---
    print(f"\n  [{model_name}] 跑验证集 (split=val) ...")
    val_metrics = model.val(
        data='data.yaml',
        imgsz=640,
        batch=32,
        split='val',
        workers=8,
        device='0',
        verbose=False,
    )
    val_txt = os.path.join(model_dir, 'val_results.txt')
    with open(val_txt, 'w', encoding='utf-8') as f:
        f.write(f"模型: {model_name}\n")
        f.write(f"权重: {best_pt_path}\n")
        f.write(f"数据集分割: val\n\n")
        f.write(format_results(val_metrics))
    print(f"  -> 验证集结果已保存: {val_txt}")

    # --- 测试集 ---
    print(f"\n  [{model_name}] 跑测试集 (split=test) ...")
    test_metrics = model.val(
        data='data.yaml',
        imgsz=640,
        batch=32,
        split='test',
        workers=8,
        device='0',
        verbose=False,
    )
    test_txt = os.path.join(model_dir, 'test_results.txt')
    with open(test_txt, 'w', encoding='utf-8') as f:
        f.write(f"模型: {model_name}\n")
        f.write(f"权重: {best_pt_path}\n")
        f.write(f"数据集分割: test\n\n")
        f.write(format_results(test_metrics))
    print(f"  -> 测试集结果已保存: {test_txt}")


if __name__ == '__main__':
    # 支持命令行指定单个模型名，否则跑全部
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target in models:
            evaluate_model(target, models[target])
        else:
            print(f"未找到模型 '{target}'，可用: {list(models.keys())}")
    else:
        for name, pt_path in sorted(models.items()):
            evaluate_model(name, pt_path)

    print("\n全部评估完成！")
