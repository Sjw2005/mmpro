"""
用法: python eval_single.py <model_name> <split>
例如: python eval_single.py DMFNet_SPDConv test
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultralytics import YOLO

CLASS_NAMES = ["car", "bus", "van", "truck", "freight_car"]
RESULTS_DIR = "results"

model_name = sys.argv[1]
split = sys.argv[2]

weight_path = os.path.join(RESULTS_DIR, model_name, "weights", "best.pt")
print(f"评估: {model_name} | split={split} | weights={weight_path}")

model = YOLO(weight_path)
metrics = model.val(data="data.yaml", split=split, batch=26, imgsz=640, device="0", plots=False, save_json=False, verbose=True)

box = metrics.box
output_filename = "val_results.txt" if split == "val" else "test_results.txt"
output_path = os.path.join(RESULTS_DIR, model_name, output_filename)
split_cn = "验证集(val)" if split == "val" else "测试集(test)"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"模型: {model_name}\n")
    f.write(f"数据集: {split_cn}\n")
    f.write(f"权重: {weight_path}\n")
    f.write(f"{'='*50}\n\n")
    f.write(f"整体表现:\n")
    f.write(f"  Precision:    {box.mp:.5f}\n")
    f.write(f"  Recall:       {box.mr:.5f}\n")
    f.write(f"  mAP@0.5:      {box.map50:.5f}\n")
    f.write(f"  mAP@0.75:     {box.map75:.5f}\n")
    f.write(f"  mAP@0.5:0.95: {box.map:.5f}\n")
    f.write(f"\n{'='*50}\n\n")
    f.write(f"各类别表现:\n")
    f.write(f"{'类别':<15} {'AP@0.5':<12} {'AP@0.5:0.95':<12}\n")
    f.write(f"{'-'*40}\n")
    for i, name in enumerate(CLASS_NAMES):
        a50 = box.ap50[i] if i < len(box.ap50) else 0
        a = box.ap[i] if i < len(box.ap) else 0
        f.write(f"{name:<15} {a50:<12.5f} {a:<12.5f}\n")

print(f"结果已保存: {output_path}")
