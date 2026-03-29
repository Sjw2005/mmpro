"""推理验证脚本：逐个加载4个改进模型做dummy forward pass，验证代码正确性。"""
import os
import sys
import torch
import warnings
warnings.filterwarnings('ignore')


def verify_model(yaml_path, model_name, env_vars=None):
    """验证单个模型能正常实例化和前向推理。"""
    print(f"\n{'='*60}")
    print(f"验证模型: {model_name}")
    print(f"配置文件: {yaml_path}")
    print(f"{'='*60}")

    # 设置环境变量
    old_env = {}
    if env_vars:
        for k, v in env_vars.items():
            old_env[k] = os.environ.get(k)
            os.environ[k] = v
            print(f"  设置环境变量: {k}={v}")

    try:
        from ultralytics import YOLO

        # 1. 实例化模型
        print(f"  [1/3] 实例化模型...")
        model = YOLO(yaml_path, task='detect')
        print(f"  ✓ 模型实例化成功")

        # 2. 打印模型信息
        print(f"  [2/3] 模型信息:")
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print(f"    总参数量: {total_params:,}")
        print(f"    可训练参数: {trainable_params:,}")

        # 3. 前向推理
        print(f"  [3/3] 前向推理测试...")
        dummy_input = torch.randn(1, 6, 640, 640)
        model.model.eval()
        with torch.no_grad():
            output = model.model(dummy_input)

        # 检查输出
        if isinstance(output, (list, tuple)):
            print(f"    输出类型: {type(output).__name__}, 包含 {len(output)} 个元素")
            for idx, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    print(f"    输出[{idx}] shape: {o.shape}")
                elif isinstance(o, (list, tuple)):
                    for jj, oo in enumerate(o):
                        if isinstance(oo, torch.Tensor):
                            print(f"    输出[{idx}][{jj}] shape: {oo.shape}")
        elif isinstance(output, torch.Tensor):
            print(f"    输出 shape: {output.shape}")

        print(f"  ✓ {model_name} 验证通过!")
        return True

    except Exception as e:
        print(f"  ✗ {model_name} 验证失败!")
        print(f"    错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 恢复环境变量
        if env_vars:
            for k in env_vars:
                if old_env.get(k) is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old_env[k]


def main():
    models = [
        {
            "name": "DMFNet_SPDConv",
            "yaml": "improve_multimodal/DMFNet_SPDConv.yaml",
            "env": {},
        },
        {
            "name": "DMFNet_DySample",
            "yaml": "improve_multimodal/DMFNet_DySample.yaml",
            "env": {},
        },
        {
            "name": "DMFNet_InnerIoU",
            "yaml": "improve_multimodal/DMFNet_InnerIoU.yaml",
            "env": {"INNER_IOU_RATIO": "0.7"},
        },
        {
            "name": "DMFNet_LSKA",
            "yaml": "improve_multimodal/DMFNet_LSKA.yaml",
            "env": {},
        },
    ]

    results = {}
    for exp in models:
        ok = verify_model(exp["yaml"], exp["name"], exp.get("env"))
        results[exp["name"]] = ok

    # 总结
    print(f"\n{'='*60}")
    print("验证总结:")
    print(f"{'='*60}")
    all_pass = True
    for name, ok in results.items():
        status = "✓ 通过" if ok else "✗ 失败"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    if all_pass:
        print(f"\n所有模型验证通过! 可以使用 MAN.py 开始训练。")
    else:
        print(f"\n存在验证失败的模型，请检查后再训练。")
        sys.exit(1)


if __name__ == "__main__":
    main()
