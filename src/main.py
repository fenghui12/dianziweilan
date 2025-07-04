#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import yaml
import time
import os
from ultralytics import YOLO
import torch
from utils import draw_fence_line, is_bbox_crossing_line, draw_warning, put_chinese_text

class ElectronicFence:
    def __init__(self, config_path='config.yaml'):
        """初始化电子围栏系统"""
        # 检测CUDA可用性
        self.device = self._check_cuda()
        print(f"使用设备: {self.device}")
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化视频源
        self.cap = self._init_video_source()
        
        # 初始化YOLO模型
        self.model = self._init_model()
        
        # 警戒线参数
        self.fence_line = self.config['fence']['line']
        self.warning_text = self.config['fence']['warning_text']
        self.warning_color = tuple(self.config['fence']['warning_color'])
        self.line_color = tuple(self.config['fence']['line_color'])
        self.line_thickness = self.config['fence']['line_thickness']
        
        # 显示参数
        self.show_fps = self.config['display']['show_fps']
        self.show_detections = self.config['display']['show_detections']
        
        # 状态变量
        self.warning_active = False
        self.fps = 0
        
    def _check_cuda(self):
        """检查CUDA可用性并返回合适的设备"""
        if torch.cuda.is_available():
            # 如果有多个GPU，显示所有可用的GPU
            if torch.cuda.device_count() > 1:
                print(f"发现 {torch.cuda.device_count()} 个GPU:")
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                print(f"发现GPU: {torch.cuda.get_device_name(0)}")
                
            # 显示CUDA版本信息
            print(f"CUDA版本: {torch.version.cuda}")
            return 'cuda:0'  # 使用第一个GPU
        else:
            print("警告: 未检测到CUDA GPU，将使用CPU进行推理")
            return 'cpu'
        
    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"配置文件加载错误: {e}")
            exit(1)
            
    def _init_video_source(self):
        """初始化视频源"""
        source = self.config['video']['source']
        try:
            # 如果是数字，则作为摄像头索引处理
            if isinstance(source, int) or source.isdigit():
                cap = cv2.VideoCapture(int(source))
            else:
                # 否则作为视频文件路径处理
                cap = cv2.VideoCapture(source)
                
            # 设置分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['video']['width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['video']['height'])
            
            if not cap.isOpened():
                raise Exception("无法打开视频源")
                
            return cap
        except Exception as e:
            print(f"视频源初始化错误: {e}")
            exit(1)
            
    def _init_model(self):
        """初始化YOLO模型"""
        try:
            # 检查模型文件是否存在
            weights_path = self.config['model']['weights']
            if not os.path.exists(weights_path):
                print(f"模型文件不存在: {weights_path}")
                print("请下载YOLOv11模型文件或修改配置")
                exit(1)
                
            # 加载模型并指定设备
            model = YOLO(weights_path).to(self.device)
            print(f"YOLOv11模型加载成功，运行于: {self.device}")
            
            # 显示模型信息
            if self.device != 'cpu':
                # 计算CUDA内存使用情况
                allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
                print(f"CUDA内存使用: {allocated:.2f} MB")
                
            return model
        except Exception as e:
            print(f"模型初始化错误: {e}")
            exit(1)
            
    def run(self):
        """运行电子围栏系统"""
        prev_time = time.time()
        processing_times = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取视频帧")
                break
                
            # 计算FPS
            current_time = time.time()
            frame_time = current_time - prev_time
            self.fps = 1 / frame_time
            prev_time = current_time
            
            # 记录处理时间用于统计
            processing_times.append(frame_time)
            if len(processing_times) > 100:
                processing_times.pop(0)
            
            # 使用YOLO检测人
            start_detect = time.time()
            results = self.model(frame, conf=self.config['model']['confidence'], classes=self.config['model']['classes'])
            detect_time = time.time() - start_detect
            
            # 重置警告状态
            self.warning_active = False
            
            # 绘制警戒线
            frame = draw_fence_line(frame, self.fence_line, self.line_color, self.line_thickness)
            
            # 处理检测结果
            if len(results) > 0:
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box[:4])
                        
                        # 检查是否越界
                        if is_bbox_crossing_line([x1, y1, x2, y2], self.fence_line, frame.shape, threshold=10):
                            self.warning_active = True
                            
                        # 显示检测结果
                        if self.show_detections:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 显示FPS和处理时间信息
            if self.show_fps:
                avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0
                fps_text = f"FPS: {int(self.fps)}"
                detect_text = f"检测时间: {detect_time*1000:.1f}ms"
                
                # 使用PIL渲染文字而不是cv2.putText
                frame = put_chinese_text(frame, fps_text, (10, 30), (0, 255, 0), 30)
                frame = put_chinese_text(frame, detect_text, (10, 70), (0, 255, 0), 20)
            
            # 如果有人越界，显示警告
            if self.warning_active:
                frame = draw_warning(frame, self.warning_text, self.warning_color)
            
            # 显示结果
            cv2.imshow("电子围栏系统", frame)
            
            # 按ESC退出
            if cv2.waitKey(1) == 27:
                break
                
        # 清理资源
        self.cap.release()
        cv2.destroyAllWindows()
        
        # 打印性能统计
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            avg_fps = 1 / avg_time
            print(f"\n性能统计:")
            print(f"平均处理时间: {avg_time*1000:.2f}ms")
            print(f"平均FPS: {avg_fps:.2f}")
        
if __name__ == "__main__":
    fence = ElectronicFence()
    fence.run() 