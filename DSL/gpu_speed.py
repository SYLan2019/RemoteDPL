# import torch
# import json
# from collections import defaultdict
# from tqdm import tqdm
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
#
# # 计算 IoU 的函数，使用向量化操作加速
# def compute_iou_tensors(boxes0, boxes1):
#     A = boxes0.size(0)
#     B = boxes1.size(0)
#
#     # 计算交集部分
#     xy_max = torch.min(boxes0[:, 2:].unsqueeze(1).expand(A, B, 2),
#                        boxes1[:, 2:].unsqueeze(0).expand(A, B, 2))
#     xy_min = torch.max(boxes0[:, :2].unsqueeze(1).expand(A, B, 2),
#                        boxes1[:, :2].unsqueeze(0).expand(A, B, 2))
#
#     inter = (xy_max - xy_min).clamp(min=0)
#     inter_area = inter[:, :, 0] * inter[:, :, 1]
#
#     # 计算每个矩形的面积
#     area_boxes0 = ((boxes0[:, 2] - boxes0[:, 0]) *
#                    (boxes0[:, 3] - boxes0[:, 1])).unsqueeze(1).expand(A, B)
#     area_boxes1 = ((boxes1[:, 2] - boxes1[:, 0]) *
#                    (boxes1[:, 3] - boxes1[:, 1])).unsqueeze(0).expand(A, B)
#
#     # 计算 IoU
#     iou = inter_area / (area_boxes0 + area_boxes1 - inter_area)
#     return iou
#
#
# # 计算一张图片中小尺度物体的重叠数
# def compute_overlaps_for_small_objects(boxes, size_threshold=32, iou_threshold=0.5):
#     """
#     计算一张图片中的小尺度物体的重叠数，使用 IoU 阈值
#     Parameters
#     ----------
#     boxes: List of bounding boxes for one image
#     size_threshold: float, area threshold to define small objects
#     iou_threshold: IoU threshold to consider as overlap
#
#     Returns
#     -------
#     overlap_count: int, number of overlapping pairs with IoU > iou_threshold
#     """
#     # 筛选面积小于 size_threshold * size_threshold 的小物体
#     small_boxes = [
#         box for box in boxes
#         if (box[2] - box[0]) * (box[3] - box[1]) < size_threshold ** 2
#     ]
#
#     if len(small_boxes) < 2:
#         return 0  # 如果小物体数小于 2，直接返回 0
#
#     # 转换成 PyTorch 张量，形状为 (N, 4)
#     boxes_tensor = torch.tensor(small_boxes, dtype=torch.float32, device='cuda')
#
#     # 计算 IoU，得到形状为 (N, N) 的 IoU 矩阵
#     iou_matrix = compute_iou_tensors(boxes_tensor, boxes_tensor)
#
#     # 忽略对角线的自身比较
#     iou_matrix = iou_matrix.triu(diagonal=1)
#
#     # 计算 IoU > 阈值的数量
#     overlap_count = (iou_matrix > iou_threshold).sum().item()
#
#     return overlap_count
#
#
# # 示例：从 COCO 数据中提取所有的边界框，并按 image_id 分组
# in_addr = '/public/home/lsy/myj/code_dsl/DSL/datalist/voc_semi/voc07_train.json'
# with open(in_addr, 'r') as f:
#     coco_data = json.load(f)
#
# # 按 image_id 分组所有边界框
# image_boxes = defaultdict(list)
# for ann in coco_data['annotations']:
#     image_id = ann['image_id']
#     bbox = ann['bbox']
#     # 将 bbox 格式从 [x_min, y_min, width, height] 转换成 [x_min, y_min, x_max, y_max]
#     bbox[2] += bbox[0]  # x_max = x_min + width
#     bbox[3] += bbox[1]  # y_max = y_min + height
#     image_boxes[image_id].append(bbox)
#
# # 设置 IoU 阈值列表
# iou_thresholds = np.arange(0.1, 1.0, 0.1)
# overlaps = []
#
# # 对不同 IoU 阈值计算重叠情况
# size_threshold = 32  # 定义小尺度物体的面积阈值
# for threshold in iou_thresholds:
#     total_overlaps = 0
#     # 计算每张图片的小尺度物体的 IoU 重叠数量
#     for image_id, boxes in tqdm(image_boxes.items(), desc=f"Calculating {int(threshold)} overlaps"):
#         overlap_count = compute_overlaps_for_small_objects(boxes, size_threshold=size_threshold,
#                                                            iou_threshold=threshold)
#         total_overlaps += overlap_count
#     print(f'IOU > {threshold} 的小尺度物体重叠数目为： {total_overlaps}')
#     overlaps.append(total_overlaps)
#
# # 绘制图表，显示 IoU 为纵坐标的重叠数量
# plt.figure(figsize=(10, 6))
# plt.plot(iou_thresholds, overlaps, marker='o', color='blue', alpha=0.7)
# plt.title("Number of Overlapping Small Object Pairs at Different IoU Thresholds")
# plt.xlabel("IoU Threshold")
# plt.ylabel("Number of Overlapping Small Object Pairs")
#
# # 保存图表
# save_path = os.path.join('/public/home/lsy/myj/code_dsl/DSL', 'small_object_overlap_analysis.png')
# plt.tight_layout()  # 调整布局，防止图像被截断
# plt.savefig(save_path)
# plt.close()  # 关闭图形，释放内存
#
# print(f"图片已保存到: {save_path}")

import torch
import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os


# 计算 IoU 的函数，使用向量化操作加速
def compute_iou_tensors(boxes0, boxes1):
    A = boxes0.size(0)
    B = boxes1.size(0)

    # 计算交集部分
    xy_max = torch.min(boxes0[:, 2:].unsqueeze(1).expand(A, B, 2),
                       boxes1[:, 2:].unsqueeze(0).expand(A, B, 2))
    xy_min = torch.max(boxes0[:, :2].unsqueeze(1).expand(A, B, 2),
                       boxes1[:, :2].unsqueeze(0).expand(A, B, 2))

    inter = (xy_max - xy_min).clamp(min=0)
    inter_area = inter[:, :, 0] * inter[:, :, 1]

    # 计算每个矩形的面积
    area_boxes0 = ((boxes0[:, 2] - boxes0[:, 0]) *
                   (boxes0[:, 3] - boxes0[:, 1])).unsqueeze(1).expand(A, B)
    area_boxes1 = ((boxes1[:, 2] - boxes1[:, 0]) *
                   (boxes1[:, 3] - boxes1[:, 1])).unsqueeze(0).expand(A, B)

    # 计算 IoU
    iou = inter_area / (area_boxes0 + area_boxes1 - inter_area)
    return iou


# 计算一张图片中物体的重叠数（可选择是否仅计算小尺度物体）
def compute_overlaps(boxes, size_threshold=None, iou_threshold=0.5):
    """
    计算一张图片中的重叠数，使用 IoU 阈值，且可选择仅计算小尺度物体
    Parameters
    ----------
    boxes: List of bounding boxes for one image
    size_threshold: float, area threshold to define small objects (if None, consider all objects)
    iou_threshold: IoU threshold to consider as overlap

    Returns
    -------
    overlap_count: int, number of overlapping pairs with IoU > iou_threshold
    """
    if size_threshold:
        # 筛选面积小于 size_threshold * size_threshold 的小物体
        boxes = [
            box for box in boxes
            if (box[2] - box[0]) * (box[3] - box[1]) < size_threshold ** 2
        ]

    if len(boxes) < 2:
        return 0  # 如果该图片中目标数小于 2，直接返回 0

    # 转换成 PyTorch 张量，形状为 (N, 4)
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device='cuda')

    # 计算 IoU，得到形状为 (N, N) 的 IoU 矩阵
    iou_matrix = compute_iou_tensors(boxes_tensor, boxes_tensor)

    # 忽略对角线的自身比较
    iou_matrix = iou_matrix.triu(diagonal=1)

    # 计算 IoU > 阈值的数量
    overlap_count = (iou_matrix > iou_threshold).sum().item()

    return overlap_count


# 示例：从 COCO 数据中提取所有的边界框，并按 image_id 分组
in_addr = '/public/home/lsy/myj/code_dsl/DSL/datalist/voc_semi/voc07_train.json'
with open(in_addr, 'r') as f:
    coco_data = json.load(f)

# 按 image_id 分组所有边界框
image_boxes = defaultdict(list)
for ann in coco_data['annotations']:
    image_id = ann['image_id']
    bbox = ann['bbox']
    # 将 bbox 格式从 [x_min, y_min, width, height] 转换成 [x_min, y_min, x_max, y_max]
    bbox[2] += bbox[0]  # x_max = x_min + width
    bbox[3] += bbox[1]  # y_max = y_min + height
    image_boxes[image_id].append(bbox)

# 设置 IoU 阈值列表
iou_thresholds = np.arange(0.1, 0.7, 0.1)
all_overlaps = []
small_overlaps = []

# 定义小尺度物体的面积阈值
size_threshold = 32

# 对不同 IoU 阈值计算所有物体和小尺度物体的重叠情况
for threshold in iou_thresholds:
    total_all_overlaps = 0
    total_small_overlaps = 0

    # 计算每张图片的 IoU 重叠数量（所有物体和小尺度物体）
    for image_id, boxes in tqdm(image_boxes.items(), desc=f"Calculating {int(threshold)} overlaps"):
        all_overlap_count = compute_overlaps(boxes, size_threshold=None, iou_threshold=threshold)
        small_overlap_count = compute_overlaps(boxes, size_threshold=size_threshold, iou_threshold=threshold)

        total_all_overlaps += all_overlap_count
        total_small_overlaps += small_overlap_count

    print(f'IOU > {threshold} 的所有物体重叠数目为： {total_all_overlaps}')
    print(f'IOU > {threshold} 的小尺度物体重叠数目为： {total_small_overlaps}')

    all_overlaps.append(total_all_overlaps)
    small_overlaps.append(total_small_overlaps)

# 绘制双柱状图进行对比
x = np.arange(len(iou_thresholds))
width = 0.45  # 设置柱状图宽度

plt.figure(figsize=(12, 7))
# 绘制所有物体的重叠情况
plt.bar(x - width / 2, all_overlaps, width, label='All Objects', color='blue', alpha=0.7)
# 绘制小尺度物体的重叠情况
plt.bar(x + width / 2, small_overlaps, width, label='Small Objects', color='orange', alpha=0.7)

# 添加图例、标题和坐标轴标签
plt.xticks(x, [f"{threshold:.1f}" for threshold in iou_thresholds], fontsize=20)
plt.yticks(fontsize = 20)
plt.xlabel("IoU Threshold", fontsize=20)
plt.ylabel("Number of Overlapping Pairs", fontsize = 20)
plt.title("Comparison of Overlapping Pairs for All and Small Objects at Different IoU Thresholds", fontsize = 18)
plt.legend(fontsize=18)

# 保存图表
save_path = os.path.join('/public/home/lsy/myj/code_dsl/DSL', 'object_overlap_comparison.png')
plt.tight_layout()  # 调整布局，防止图像被截断
plt.savefig(save_path)
plt.close()  # 关闭图形，释放内存

print(f"图片已保存到: {save_path}")