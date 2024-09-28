from pathlib import Path
from ultralytics import SAM, YOLO
import torch
import matplotlib.pyplot as plt
import cv2

# 定义图像数据路径
img_data_path = 'ultralytics/videos'

# 定义检测模型和SAM模型的路径
det_model = "vision.pt"
sam_model = "MyModels/transfered_sam.pt"

# 根据CUDA是否可用选择设备
device = '0' if torch.cuda.is_available() else 'cpu'

# 初始化检测模型和SAM模型
det_model = YOLO(det_model)
sam_model = SAM(sam_model)

# 获取图像数据路径
data = Path(img_data_path)

# 定义根输出目录
output_root_dir = data.parent / f"{data.stem}_auto_annotate"
# 创建根输出目录
output_root_dir.mkdir(exist_ok=True, parents=True)

# 对图像数据进行检测
det_results = det_model(data, stream=True, device=device)

# 遍历检测结果
for result in det_results:
    # 获取图像名称
    img_name = Path(result.path).stem
    # 为每个图像创建独立的输出文件夹
    output_dir = output_root_dir / img_name
    output_dir.mkdir(exist_ok=True, parents=True)

    # 保存原始图像到对应文件夹
    img_output_path = output_dir / f"{img_name}.jpg"
    cv2.imwrite(str(img_output_path), result.orig_img)

    # 获取类别ID
    class_ids = result.boxes.cls.int().tolist()  # noqa
    # 如果有检测到物体
    if len(class_ids):
        # 获取检测框坐标
        boxes = result.boxes.xyxy  # Boxes object for bbox outputs
        # 使用SAM模型进行分割
        sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)
        # 获取分割结果
        segments = sam_results[0].masks.xyn  # noqa

        # 绘制分割结果并保存
        img = result.orig_img
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        for i, segment in enumerate(segments):
            if len(segment) == 0:
                continue
            segment = segment * [img.shape[1], img.shape[0]]  # 归一化坐标转换为图像坐标
            ax.fill(*zip(*segment), alpha=0.5, label=f'Class {class_ids[i]}')
        ax.legend()
        ax.axis('off')

        # 保存带有分割结果的图像
        seg_output_path = output_dir / f"{img_name}_seg.png"
        fig.savefig(seg_output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # 为每个图像生成标注文件
        with open(output_dir / f"{img_name}.txt", "w") as f:
            # 遍历每个分割区域
            for i in range(len(segments)):
                s = segments[i]
                # 如果分割区域为空，则跳过
                if len(s) == 0:
                    continue
                # 将分割区域坐标转换为字符串格式
                segment = map(str, segments[i].reshape(-1).tolist())
                # 写入标注信息
                f.write(f"{class_ids[i]} " + " ".join(segment) + "\n")

# 提示处理完成
print("所有图像的目标检测与分割结果已保存。")
