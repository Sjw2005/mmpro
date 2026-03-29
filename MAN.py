"""类别加权BCE对比实验：轻度/中度/重度三组权重。"""
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 可配置参数
# ============================================================
BATCH_SIZE = 32
EPOCHS = 150
IMG_SIZE = 640
DEVICE = '0'
WORKERS = 8
DATA_YAML = 'data.yaml'
PROJECT = 'results'

# 基础模型配置 (使用当前最优的 MDAFP3D)
BASE_YAML = "improve_multimodal/DMFNet_MDAFP3D.yaml"

# ============================================================
# 类别加权实验列表
# 类别顺序: 0:car, 1:bus, 2:van, 3:truck, 4:freight_car
# 频率:     85.9%  5.4%  2.8%  2.7%   3.2%
# ============================================================
experiments = [
    {
        "name": "DMFNet_MDAFP3D_ClsW_Light",
        "yaml": BASE_YAML,
        "env": {"CLS_WEIGHTS": "1.0,2.0,2.5,2.8,2.6"},
        "desc": "轻度加权: car=1.0, bus=2.0, van=2.5, truck=2.8, freight=2.6",
    },
    {
        "name": "DMFNet_MDAFP3D_ClsW_Medium",
        "yaml": BASE_YAML,
        "env": {"CLS_WEIGHTS": "1.0,3.0,4.0,4.2,3.8"},
        "desc": "中度加权: car=1.0, bus=3.0, van=4.0, truck=4.2, freight=3.8",
    },
    {
        "name": "DMFNet_MDAFP3D_ClsW_Heavy",
        "yaml": BASE_YAML,
        "env": {"CLS_WEIGHTS": "1.0,4.0,5.5,5.6,5.2"},
        "desc": "重度加权: car=1.0, bus=4.0, van=5.5, truck=5.6, freight=5.2",
    },
]


def run_experiment(exp):
    """运行单个实验。"""
    name = exp["name"]
    yaml_path = exp["yaml"]
    env_vars = exp.get("env", {})
    desc = exp.get("desc", "")

    print(f"\n{'#'*60}")
    print(f"# 开始实验: {name}")
    print(f"# 说明: {desc}")
    print(f"# 配置: {yaml_path}")
    print(f"# Batch Size: {BATCH_SIZE}")
    print(f"{'#'*60}\n")

    # 设置环境变量
    old_env = {}
    for k, v in env_vars.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = v
        print(f"  设置环境变量: {k}={v}")

    try:
        from ultralytics import YOLO

        model = YOLO(yaml_path, task='detect')
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT,
            name=name,
            exist_ok=True,
            pretrained=False,
            amp=False,
        )
        print(f"\n  实验 {name} 训练完成!")

    except Exception as e:
        print(f"\n  实验 {name} 训练失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 恢复环境变量
        for k in env_vars:
            if old_env.get(k) is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_env[k]


if __name__ == '__main__':
    print("="*60)
    print("类别加权BCE对比实验 (基于DMFNet_MDAFP3D)")
    print(f"Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS} | ImgSize: {IMG_SIZE}")
    print("="*60)
    print("\n实验计划:")
    for i, exp in enumerate(experiments):
        print(f"  [{i+1}] {exp['name']} - {exp['desc']}")

    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {exp['name']}")
        run_experiment(exp)

    print(f"\n{'='*60}")
    print("所有实验完成!")
    print(f"{'='*60}")

    os.system('shutdown')
