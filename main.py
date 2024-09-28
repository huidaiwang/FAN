from pathlib import Path
from ultralytics import SAM, YOLO
import torch
import matplotlib.pyplot as plt
import cv2
import ollama

# 初始化模型
det_model = YOLO("vision.pt")
sam_model = SAM("MyModels/transfered_sam.pt")
device = '0' if torch.cuda.is_available() else 'cpu'

# 定义路径
video_path = 'ultralytics/videos/test.mp4'
output_root_dir = Path('output')
output_root_dir.mkdir(exist_ok=True, parents=True)

# 车道与车流量统计
lane_counts = {}  # 车道统计结果

def api_generate(text: str):
    print(f'提问：{text}')
    stream = ollama.generate(stream=True, model='hhui/fan1.0', prompt=text)
    response = ""
    for chunk in stream:
        if not chunk['done']:
            response += chunk['response']
        else:
            print('\n')
            print(f'总耗时：{chunk["total_duration"]}')
            return response
    return response

def process_frame(frame, frame_id):
    # 目标检测
    result = det_model(frame, device=device)
    img_name = f"frame_{frame_id}"

    # 保存原始图像
    img_output_path = output_root_dir / f"{img_name}.jpg"
    cv2.imwrite(str(img_output_path), frame)

    # 获取检测框
    boxes = result[0].boxes.xyxy
    class_ids = result[0].boxes.cls.int().tolist()
    if len(class_ids):
        # 使用SAM模型进行分割
        sam_results = sam_model(frame, bboxes=boxes, device=device)
        segments = sam_results[0].masks.xyn

        # 保存分割图像
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(frame)
        for i, segment in enumerate(segments):
            if len(segment) == 0:
                continue
            segment = segment * [frame.shape[1], frame.shape[0]]
            ax.fill(*zip(*segment), alpha=0.5, label=f'Class {class_ids[i]}')
        ax.legend()
        ax.axis('off')
        seg_output_path = output_root_dir / f"{img_name}_seg.png"
        fig.savefig(seg_output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # 保存标注文件
        with open(output_root_dir / f"{img_name}.txt", "w") as f:
            for i in range(len(segments)):
                s = segments[i]
                if len(s) == 0:
                    continue
                segment = map(str, segments[i].reshape(-1).tolist())
                f.write(f"{class_ids[i]} " + " ".join(segment) + "\n")

        # 调用多模态模型进行内容推理
        prompt = f"车道检测：请根据以下检测结果估计车道数量和每个车道的车流量。检测框：{boxes.tolist()}"
        response = api_generate(prompt)
        print(f"模型响应:\n{response}")

        # 更新车道统计（这里是一个简单示例，你可以根据模型的具体输出更新统计）
        lanes = response.split()  # 根据模型响应解析车道数量和车流量
        for lane in lanes:
            lane_counts[lane] = lane_counts.get(lane, 0) + 1

# 读取视频并逐帧处理
cap = cv2.VideoCapture(video_path)
frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    process_frame(frame, frame_id)
    frame_id += 1

cap.release()

# 输出车道统计结果
print("车道统计结果：")
for lane, count in lane_counts.items():
    print(f"车道 {lane}: {count} 辆车")

print("所有图像的目标检测与分割结果已保存。")
