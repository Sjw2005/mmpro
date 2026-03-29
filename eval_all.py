"""
对 results/ 下所有模型的 best.pt 分别在验证集和测试集上评估，
在每个模型文件夹中生成 val_results.txt 和 test_results.txt
"""
import os
import sys

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO

RESULTS_DIR = "results"
CLASS_NAMES = ["car", "bus", "van", "truck", "freight_car"]

MODELS = [
    "DMFNet_SPDConv",
    "DMFNet_DySample",
    "DMFNet_InnerIoU",
    "DMFNet_LSKA",
    "DMFNet_MDAFP3D",
    "DMFNet_MDAFP3D_128",
    "DMFNet_MDAFP3D_Conv1x1",
]


def eval_and_save(model_name, split, output_filename):
    """加载 best.pt 并在指定 split 上评估，结果写入 txt"""
    weight_path = os.path.join(RESULTS_DIR, model_name, "weights", "best.pt")
    if not os.path.exists(weight_path):
        print(f"[SKIP] {weight_path} 不存在")
        return

    print(f"\n{'='*60}")
    print(f"评估: {model_name} | split={split}")
    print(f"{'='*60}")

    model = YOLO(weight_path)
    metrics = model.val(
        data="data.yaml",
        split=split,
        batch=26,
        imgsz=640,
        device="0",
        plots=False,
        save_json=False,
        verbose=True,
    )

    # 提取指标
    box = metrics.box
    # 整体指标
    mp = box.mp        # mean precision
    mr = box.mr        # mean recall
    map50 = box.map50  # mAP@0.5
    map75 = box.map75  # mAP@0.75
    map_val = box.map  # mAP@0.5:0.95

    # 各类别指标
    ap50_per_class = box.ap50       # shape (nc,)
    ap_per_class = box.ap           # shape (nc,) mAP50-95

    # 写入文件
    output_path = os.path.join(RESULTS_DIR, model_name, output_filename)
    split_cn = "验证集(val)" if split == "val" else "测试集(test)"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"模型: {model_name}\n")
        f.write(f"数据集: {split_cn}\n")
        f.write(f"权重: {weight_path}\n")
        f.write(f"{'='*50}\n\n")

        f.write(f"整体表现:\n")
        f.write(f"  Precision:    {mp:.5f}\n")
        f.write(f"  Recall:       {mr:.5f}\n")
        f.write(f"  mAP@0.5:      {map50:.5f}\n")
        f.write(f"  mAP@0.75:     {map75:.5f}\n")
        f.write(f"  mAP@0.5:0.95: {map_val:.5f}\n")
        f.write(f"\n{'='*50}\n\n")

        f.write(f"各类别表现:\n")
        f.write(f"{'类别':<15} {'AP@0.5':<12} {'AP@0.5:0.95':<12}\n")
        f.write(f"{'-'*40}\n")
        for i, name in enumerate(CLASS_NAMES):
            a50 = ap50_per_class[i] if i < len(ap50_per_class) else 0
            a = ap_per_class[i] if i < len(ap_per_class) else 0
            f.write(f"{name:<15} {a50:<12.5f} {a:<12.5f}\n")

    print(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    for model_name in MODELS:
        eval_and_save(model_name, "val", "val_results.txt")
        eval_and_save(model_name, "test", "test_results.txt")

    print("\n\n全部评估完成!")
