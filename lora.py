from ultralytics import RTDETR,YOLO
from ultralytics.nn import AddModules
import warnings

warnings.filterwarnings('ignore')
# 实时保存日志，断网可训↓↓↓  
# 终端运行命令：nohup python train.py > logs/DMFNet_Finetune.txt 2>&1 & echo $! > logs/train_finetune.pid && tail -f logs/DMFNet_Finetune.txt

model = YOLO("results/DMFNet_MDAFP3D/weights/best.pt")

model.train(
    task='detect',
    mode='train',
    data='data.yaml',
    
    epochs=80,
    time=None,
    patience=30,
    
    batch=24,
    imgsz=640,
    
    save=True,
    save_period=-1,
    cache=False,
    device='0',
    workers=4,
    optimizer='auto',
    verbose=True,
    seed=0,
    deterministic=True,
    single_cls=False,
    rect=False,
    
    cos_lr=True,
    close_mosaic=10,
    resume=False,
    amp=False,
    fraction=1.0,
    profile=False,
    freeze=None,
    multi_scale=False,
    overlap_mask=True,
    mask_ratio=4,
    dropout=0.0,
    val=True,
    split='val',
    save_json=False,
    save_hybrid=False,
    conf=None,
    iou=0.7,
    max_det=300,
    half=False,
    dnn=False,
    plots=True,
    source=None,
    vid_stride=1,
    stream_buffer=False,
    visualize=True,
    augment=False,
    agnostic_nms=False,
    classes=None,
    retina_masks=False,
    embed=None,
    show=False,
    save_frames=False,
    save_txt=False,
    save_conf=False,
    save_crop=False,
    show_labels=True,
    show_conf=True,
    show_boxes=True,
    line_width=1,
    format='torchscript',
    keras=False,
    optimize=False,
    int8=False,
    dynamic=False,
    simplify=True,
    opset=None,
    workspace=4,
    nms=False,
    
    # 🔴 微调学习率要下降
    lr0=0.005,         # 🌟 优化：从 0.01 降为 0.005
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    box=7.5,
    cls=1.5,
    dfl=1.5,
    pose=12.0,
    kobj=1.0,
    label_smoothing=0.0,
    nbs=64,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    
    degrees=180.0,     # 🌟 优化：从 0.0 改为 180.0，适应航拍下的任意车辆朝向
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.5,        # 🌟 优化：增加上下翻转 (从 0.0 改为 0.5)
    fliplr=0.5,
    bgr=0.0,
    mosaic=0.5,
    mixup=0.0,
    copy_paste=0.0,
    copy_paste_mode='flip',
    auto_augment='randaugment',
    erasing=0.0,
    crop_fraction=1.0,
    cfg=None,
    tracker='botsort.yaml',
    
    project="results",
    name="DMFNet_MDAFP3D_Finetune_960", 
    exist_ok=True,
    pretrained=False,
)
print("✅ 模型微调训练完成！")