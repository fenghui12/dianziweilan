#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import yaml
from typing import Tuple, List, Union
from PIL import Image, ImageDraw, ImageFont
import time

# 加载配置文件
def load_config(config_path='config.yaml'):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"配置文件加载错误: {e}")
        return None

# 尝试加载配置
CONFIG = load_config()
FONT = None
FONT_SIZE = 30

# 如果配置加载成功，初始化字体
if CONFIG and 'display' in CONFIG:
    try:
        font_name = CONFIG['display'].get('font', 'SimHei')
        FONT_SIZE = CONFIG['display'].get('font_size', 30)
        # 尝试加载系统字体
        FONT = ImageFont.truetype(font_name, FONT_SIZE)
    except Exception as e:
        print(f"字体加载错误: {e}")
        print("尝试使用默认字体")
        try:
            # 尝试使用默认字体
            FONT = ImageFont.load_default()
        except:
            print("无法加载默认字体，将使用OpenCV原生文字渲染")

def draw_fence_line(frame, line, color, thickness):
    """绘制警戒线"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = line
    
    # 将归一化坐标转换为实际像素坐标
    x1, y1 = int(x1 * w), int(y1 * h)
    x2, y2 = int(x2 * w), int(y2 * h)
    
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    return frame

def draw_fence_area(frame, area_points, color, thickness, fill_color=None):
    """
    绘制警戒区域
    
    参数:
    - frame: 视频帧
    - area_points: 归一化坐标的区域顶点列表 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    - color: 边界线颜色
    - thickness: 边界线粗细
    - fill_color: 填充颜色 (可选，带透明度)
    
    返回:
    - 绘制好的帧
    """
    h, w = frame.shape[:2]
    
    # 将归一化坐标转换为实际像素坐标
    points = []
    for x, y in area_points:
        points.append((int(x * w), int(y * h)))
    
    points = np.array(points, np.int32)
    points = points.reshape((-1, 1, 2))
    
    # 如果需要填充区域
    if fill_color:
        # 创建透明覆盖层
        overlay = frame.copy()
        
        # 使用fillPoly填充多边形
        r, g, b, a = fill_color
        cv2.fillPoly(overlay, [points], (b, g, r))
        
        # 应用透明度
        alpha = a / 255.0
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # 绘制边界线
    cv2.polylines(frame, [points], True, color, thickness)
    
    return frame

def is_point_in_polygon(point, polygon):
    """
    检查点是否在多边形内部
    
    参数:
    - point: [x, y] 点的坐标
    - polygon: [[x1,y1], [x2,y2], ...] 多边形顶点坐标
    
    返回:
    - True: 点在多边形内或边上
    - False: 点在多边形外
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= x_intersect:
                    inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def is_bbox_crossing_line(bbox, line, frame_shape, threshold=0):
    """检查边界框是否越过警戒线"""
    h, w = frame_shape[:2]
    x1_line, y1_line, x2_line, y2_line = line
    
    # 将归一化坐标转换为实际像素坐标
    x1_line, y1_line = int(x1_line * w), int(y1_line * h)
    x2_line, y2_line = int(x2_line * w), int(y2_line * h)
    
    # 获取边界框的底部中点（通常是人的脚部位置）
    x_min, y_min, x_max, y_max = bbox
    foot_point = [(x_min + x_max) // 2, y_max]
    
    # 计算点到线段的距离
    def point_to_line_dist(point, line_point1, line_point2):
        x0, y0 = point
        x1, y1 = line_point1
        x2, y2 = line_point2
        
        # 计算线段长度的平方
        line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        
        # 如果线段长度为0，则返回点到端点的距离
        if line_len_sq == 0:
            return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        
        # 计算投影比例 t
        t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / line_len_sq))
        
        # 计算投影点
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        # 返回点到投影点的距离
        return np.sqrt((x0 - proj_x) ** 2 + (y0 - proj_y) ** 2)
    
    # 计算脚部中点到警戒线的距离
    dist = point_to_line_dist(foot_point, (x1_line, y1_line), (x2_line, y2_line))
    
    # 如果距离小于阈值，认为越线
    return dist <= threshold

def is_bbox_in_area(bbox, area_points, frame_shape):
    """
    检查人员是否在警戒区域内
    使用人物边界框底部中心点作为判断依据
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    
    # 计算人物底部中心点（脚部位置）
    foot_x = (x1 + x2) / 2
    foot_y = y2
    
    # 将区域点转换为实际像素坐标
    area = []
    for x, y in area_points:
        area.append((int(x * w), int(y * h)))
    
    # 使用cv2.pointPolygonTest检查点是否在多边形内部
    point = (int(foot_x), int(foot_y))
    
    # 转换为numpy数组格式
    area_np = np.array(area, np.int32)
    result = cv2.pointPolygonTest(area_np, point, False)
    
    # 如果结果大于等于0，说明点在多边形内部或边界上
    return result >= 0, point

def draw_judgment_point(frame, point, in_area=False):
    """
    绘制判断点（脚部中心位置）
    
    参数:
    - frame: 视频帧
    - point: 判断点坐标 (x, y)
    - in_area: 是否在警戒区域内
    
    返回:
    - 绘制好的帧
    """
    # 根据是否在区域内选择颜色
    color = (0, 255, 0) if not in_area else (0, 0, 255)  # 绿色表示安全，红色表示危险
    
    # 绘制十字点
    x, y = point
    cv2.line(frame, (x-7, y), (x+7, y), color, 2)
    cv2.line(frame, (x, y-7), (x, y+7), color, 2)
    
    # 绘制圆形
    cv2.circle(frame, point, 5, color, -1)
    
    return frame

def find_nearest_point(cursor_pos, area_points, frame_shape, threshold=10):
    """
    找到离光标最近的区域顶点
    
    参数:
    - cursor_pos: 光标位置 (x, y)
    - area_points: 归一化坐标的区域顶点列表
    - frame_shape: 帧尺寸 (h, w, c)
    - threshold: 选中阈值（像素）
    
    返回:
    - 最近顶点的索引，如果没有在阈值内找到则返回-1
    """
    h, w = frame_shape[:2]
    cx, cy = cursor_pos
    
    # 计算每个点到光标的距离
    min_dist = float('inf')
    min_idx = -1
    
    for i, (nx, ny) in enumerate(area_points):
        # 转换为实际像素坐标
        px, py = int(nx * w), int(ny * h)
        
        # 计算欧几里得距离
        dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        
        # 更新最小距离
        if dist < min_dist and dist < threshold:
            min_dist = dist
            min_idx = i
    
    return min_idx

def draw_warning(frame, text, color=(0, 0, 255)):
    """绘制警告文本"""
    h, w = frame.shape[:2]
    
    # 创建半透明红色覆盖层
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
    
    # 应用透明度
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # 使用PIL绘制警告文本
    frame = put_chinese_text(frame, text, (w//2, h//2), color, 50, center=True)
    
    return frame

def put_chinese_text(img, text, position, color, size, center=False):
    """使用PIL绘制中文文本"""
    if isinstance(img, np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
    else:
        img_pil = img
        
    # 选择字体
    try:
        font = ImageFont.truetype("simhei.ttf", size)
    except IOError:
        # 使用默认字体
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(img_pil)
    
    # 如果需要居中，计算文本宽度和高度
    if center:
        # 更新：使用textbbox代替已弃用的textsize
        try:
            # 在较新的Pillow版本中使用textbbox
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except AttributeError:
            # 向后兼容：在旧版本中尝试使用textsize
            try:
                tw, th = draw.textsize(text, font=font)
            except:
                # 最后的后备：使用估计值
                tw, th = len(text) * size // 2, size
                
        position = (position[0] - tw // 2, position[1] - th // 2)
    
    # 绘制文本
    draw.text(position, text, font=font, fill=color)
    
    # 将PIL图像转换回OpenCV格式
    if isinstance(img, np.ndarray):
        img_opencv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_opencv
    else:
        return img_pil

class ByteTrack:
    """
    简化版的ByteTrack跟踪器
    用于跟踪检测到的目标并分配唯一ID
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        初始化跟踪器
        
        参数:
        - max_age: 轨迹可以存活的最大帧数（未匹配）
        - min_hits: 确认轨迹所需的最小匹配数
        - iou_threshold: IOU匹配阈值
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks = []  # 活动轨迹列表
        
    def _calculate_iou(self, box1, box2):
        """
        计算两个边界框的IOU
        
        参数:
        - box1, box2: [x1, y1, x2, y2] 格式的边界框
        """
        # 计算交集区域
        xx1 = max(box1[0], box2[0])
        yy1 = max(box1[1], box2[1])
        xx2 = min(box1[2], box2[2])
        yy2 = min(box1[3], box2[3])
        
        # 计算交集面积
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        intersection = w * h
        
        # 计算并集面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        # 计算IOU
        iou = intersection / union if union > 0 else 0
        return iou
    
    def update(self, detections):
        """
        更新跟踪器
        
        参数:
        - detections: 检测结果列表，每个元素为 [x1, y1, x2, y2]
        
        返回:
        - 带有ID的跟踪结果列表，每个元素为 [track_id, x1, y1, x2, y2]
        """
        # 如果没有活动轨迹，初始化所有检测为新轨迹
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append({
                    'id': self.next_id,
                    'box': det,
                    'age': 1,
                    'hits': 1,
                    'time_since_update': 0
                })
                self.next_id += 1
            return self._get_tracks()
        
        # 如果没有检测，更新所有轨迹的age
        if len(detections) == 0:
            for track in self.tracks:
                track['age'] += 1
                track['time_since_update'] += 1
            
            # 删除过期的轨迹
            self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
            return self._get_tracks()
        
        # 计算IoU矩阵
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self._calculate_iou(track['box'], det)
        
        # 使用匈牙利算法进行匹配
        matched_indices = self._hungarian_matching(iou_matrix)
        
        # 更新匹配的轨迹
        for track_idx, det_idx in matched_indices:
            if iou_matrix[track_idx, det_idx] >= self.iou_threshold:
                self.tracks[track_idx]['box'] = detections[det_idx]
                self.tracks[track_idx]['hits'] += 1
                self.tracks[track_idx]['age'] += 1
                self.tracks[track_idx]['time_since_update'] = 0
            else:
                # 匹配质量不好，视为未匹配
                self.tracks[track_idx]['age'] += 1
                self.tracks[track_idx]['time_since_update'] += 1
        
        # 找出未匹配的轨迹和检测
        unmatched_tracks = [t for t in range(len(self.tracks)) if t not in [i[0] for i in matched_indices]]
        unmatched_detections = [d for d in range(len(detections)) if d not in [i[1] for i in matched_indices]]
        
        # 更新未匹配的轨迹
        for idx in unmatched_tracks:
            self.tracks[idx]['age'] += 1
            self.tracks[idx]['time_since_update'] += 1
        
        # 创建新的轨迹
        for idx in unmatched_detections:
            self.tracks.append({
                'id': self.next_id,
                'box': detections[idx],
                'age': 1,
                'hits': 1,
                'time_since_update': 0
            })
            self.next_id += 1
        
        # 删除过期的轨迹
        self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
        
        return self._get_tracks()
    
    def _hungarian_matching(self, iou_matrix):
        """
        使用贪婪算法进行匹配（替代完整的匈牙利算法）
        
        参数:
        - iou_matrix: IoU矩阵
        
        返回:
        - 匹配索引列表，每个元素为 (track_idx, detection_idx)
        """
        # 简化版贪婪匹配
        matched_indices = []
        
        # 按IoU值从大到小排序
        flat_iou = iou_matrix.flatten()
        indices = np.argsort(-flat_iou)
        
        track_indices = set()
        det_indices = set()
        
        rows, cols = iou_matrix.shape
        for idx in indices:
            if flat_iou[idx] < self.iou_threshold:
                break
                
            row = idx // cols
            col = idx % cols
            
            if row not in track_indices and col not in det_indices:
                matched_indices.append((row, col))
                track_indices.add(row)
                det_indices.add(col)
        
        return matched_indices
    
    def _get_tracks(self):
        """
        获取当前活动的轨迹
        
        返回:
        - 带有ID的跟踪结果列表，每个元素为 [track_id, x1, y1, x2, y2]
        """
        results = []
        for track in self.tracks:
            # 只返回确认的轨迹
            if track['hits'] >= self.min_hits or track['age'] <= 2:
                results.append([
                    track['id'],
                    track['box'][0],
                    track['box'][1],
                    track['box'][2],
                    track['box'][3]
                ])
        return results 