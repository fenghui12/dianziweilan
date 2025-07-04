#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import yaml
import time
import os
from ultralytics import YOLO
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QPushButton, 
                            QCheckBox, QGroupBox, QMessageBox, QSlider,
                            QSpinBox, QDoubleSpinBox)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QMouseEvent
from PyQt5.QtCore import Qt, QTimer
from utils import (draw_fence_line, draw_fence_area, is_bbox_crossing_line, 
                  draw_warning, put_chinese_text, is_bbox_in_area, 
                  find_nearest_point, draw_judgment_point, ByteTrack)

class ElectronicFenceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置窗口标题和大小
        self.setWindowTitle("电子围栏监控系统")
        self.setGeometry(100, 100, 1000, 700)
        
        # 检测CUDA可用性
        self.device = self._check_cuda()
        
        # 初始化参数
        self.config_path = 'config.yaml'
        self.config = self._load_config(self.config_path)
        self.cap = None
        self.model = None
        self.is_running = False
        self.camera_list = self._get_available_cameras()
        self.current_camera = 0
        
        # 警戒区域参数
        if 'area' in self.config['fence'] and len(self.config['fence']['area']) >= 4:
            self.fence_area = self.config['fence']['area']
        else:
            # 默认警戒区域 - 使用线转换为四边形区域
            line = self.config['fence']['line']
            offset = 0.03  # 线两侧偏移量
            self.fence_area = [
                [line[0] - offset, line[1]],  # 左上
                [line[0] + offset, line[1]],  # 右上
                [line[2] + offset, line[3]],  # 右下
                [line[2] - offset, line[3]]   # 左下
            ]
            
        self.warning_text = self.config['fence']['warning_text']
        self.warning_color = tuple(self.config['fence']['warning_color'])
        self.line_color = tuple(self.config['fence']['line_color'])
        self.line_thickness = self.config['fence']['line_thickness']
        
        # 显示参数
        self.show_fps = self.config['display']['show_fps']
        self.show_detections = self.config['display']['show_detections']
        self.show_judgment_point = True  # 显示判断点
        
        # 警报触发参数
        self.warning_threshold = 2.0  # 默认2秒触发警报
        self.intrusion_start_time = None  # 记录入侵开始时间
        self.warning_active = False  # 警报是否激活
        
        # 编辑模式参数
        self.edit_mode = False  # 是否在编辑模式
        self.preview_mode = False  # 是否在预览模式（不进行检测）
        self.selected_point = -1  # 选中的顶点索引
        self.original_frame = None  # 存储原始帧用于编辑模式
        
        # 状态变量
        self.fps = 0
        self.processing_times = []
        
        # 目标跟踪器
        self.tracker = ByteTrack(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # 跟踪状态字典 - 记录每个ID的入侵状态和时间
        self.track_status = {}  # {id: {'intrusion_start': timestamp, 'warning': bool}}
        
        # 设置UI
        self._init_ui()
        
        # 定时器，用于更新视频帧
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.prev_time = time.time()
        
    def _check_cuda(self):
        """检查CUDA可用性并返回合适的设备"""
        if torch.cuda.is_available():
            print(f"发现GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
            return 'cuda:0'
        else:
            print("警告: 未检测到CUDA GPU，将使用CPU进行推理")
            return 'cpu'
        
    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"配置文件加载错误: {e}")
            sys.exit(1)
            
    def _get_available_cameras(self):
        """获取可用的摄像头列表"""
        camera_list = []
        # 检测系统中最多10个摄像头设备
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_name = f"摄像头 {i}"
                ret, _ = cap.read()
                if ret:
                    camera_list.append((i, camera_name))
                cap.release()
        
        # 如果没有找到摄像头，添加警告
        if not camera_list:
            print("警告：未检测到可用摄像头")
            
        return camera_list
        
    def _init_ui(self):
        """初始化用户界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 顶部控制面板
        control_panel = QHBoxLayout()
        
        # 摄像头选择组
        camera_group = QGroupBox("摄像头选择")
        camera_layout = QVBoxLayout()
        
        self.camera_combo = QComboBox()
        for camera_id, camera_name in self.camera_list:
            self.camera_combo.addItem(camera_name, camera_id)
        
        # 添加视频文件选项
        self.camera_combo.addItem("视频文件", "video_file")
        
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        camera_layout.addWidget(self.camera_combo)
        
        # 添加视频文件路径显示
        self.video_path_label = QLabel("选择视频文件...")
        self.video_path_label.setVisible(False)
        camera_layout.addWidget(self.video_path_label)
        
        # 添加视频文件选择按钮
        self.browse_button = QPushButton("浏览...")
        self.browse_button.clicked.connect(self._browse_video_file)
        self.browse_button.setVisible(False)
        camera_layout.addWidget(self.browse_button)
        
        camera_group.setLayout(camera_layout)
        control_panel.addWidget(camera_group)
        
        # 检测控制组
        detection_group = QGroupBox("检测控制")
        detection_layout = QVBoxLayout()
        
        # 开始/停止按钮
        self.start_stop_button = QPushButton("开始监控")
        self.start_stop_button.clicked.connect(self._toggle_monitoring)
        detection_layout.addWidget(self.start_stop_button)
        
        # 显示检测框复选框
        self.show_detection_box = QCheckBox("显示检测框")
        self.show_detection_box.setChecked(self.show_detections)
        self.show_detection_box.stateChanged.connect(self._toggle_detection_box)
        detection_layout.addWidget(self.show_detection_box)
        
        # 显示判断点复选框
        self.show_judgment_point_box = QCheckBox("显示判断点")
        self.show_judgment_point_box.setChecked(self.show_judgment_point)
        self.show_judgment_point_box.stateChanged.connect(self._toggle_judgment_point)
        detection_layout.addWidget(self.show_judgment_point_box)
        
        # 显示FPS复选框
        self.show_fps_box = QCheckBox("显示FPS")
        self.show_fps_box.setChecked(self.show_fps)
        self.show_fps_box.stateChanged.connect(self._toggle_fps)
        detection_layout.addWidget(self.show_fps_box)
        
        detection_group.setLayout(detection_layout)
        control_panel.addWidget(detection_group)
        
        # 警戒区域控制组
        fence_group = QGroupBox("警戒区域控制")
        fence_layout = QVBoxLayout()
        
        # 编辑警戒区域按钮
        self.edit_area_button = QPushButton("编辑警戒区域")
        self.edit_area_button.setCheckable(True)
        self.edit_area_button.clicked.connect(self._toggle_edit_mode)
        fence_layout.addWidget(self.edit_area_button)
        
        # 警报触发时间阈值
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("警报触发时间阈值(秒):"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 10.0)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setValue(self.warning_threshold)
        self.threshold_spin.valueChanged.connect(self._update_threshold)
        threshold_layout.addWidget(self.threshold_spin)
        fence_layout.addLayout(threshold_layout)
        
        # 保存配置按钮
        self.save_config_button = QPushButton("保存警戒区域设置")
        self.save_config_button.clicked.connect(self._save_config)
        fence_layout.addWidget(self.save_config_button)
        
        fence_group.setLayout(fence_layout)
        control_panel.addWidget(fence_group)
        
        # 将控制面板添加到主布局
        main_layout.addLayout(control_panel)
        
        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("点击「编辑警戒区域」按钮开始预览和编辑")
        self.video_label.setStyleSheet("background-color: black; color: white;")
        
        # 添加鼠标事件处理
        self.video_label.mousePressEvent = self._on_video_click
        
        main_layout.addWidget(self.video_label)
        
        # 状态栏
        self.statusBar().showMessage(f"系统已准备就绪，运行于: {self.device}")
    
    def _toggle_judgment_point(self, state):
        """切换是否显示判断点"""
        self.show_judgment_point = (state == Qt.Checked)
    
    def _on_camera_changed(self, index):
        """处理摄像头选择变化"""
        # 如果正在运行，先停止
        if self.is_running or self.preview_mode:
            self._stop_camera()
        
        # 获取选定的摄像头
        if index < len(self.camera_list):
            self.current_camera = self.camera_list[index][0]
            self.video_path_label.setVisible(False)
            self.browse_button.setVisible(False)
        else:
            # 选择了视频文件
            self.current_camera = "video_file"
            self.video_path_label.setVisible(True)
            self.browse_button.setVisible(True)
        
        # 如果在编辑模式中，重新启动摄像头预览
        if self.edit_mode:
            self._start_preview()
    
    def _browse_video_file(self):
        """浏览选择视频文件"""
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mkv *.mov);;所有文件 (*)"
        )
        
        if file_path:
            self.video_path_label.setText(file_path)
            self.current_camera = file_path
            
            # 如果在编辑模式中，重新启动摄像头预览
            if self.edit_mode:
                self._start_preview()
    
    def _toggle_monitoring(self):
        """开始/停止监控"""
        if not self.is_running:
            # 如果在预览模式，先停止预览
            if self.preview_mode:
                self._stop_camera()
                
            self._start_monitoring()
        else:
            self._stop_monitoring()
    
    def _toggle_edit_mode(self, checked):
        """切换警戒区域编辑模式"""
        self.edit_mode = checked
        
        if checked:
            # 进入编辑模式
            self.edit_area_button.setText("完成编辑")
            
            # 如果正在监控，先停止
            if self.is_running:
                self._stop_monitoring()
            
            # 启动预览模式
            self._start_preview()
            
            self.statusBar().showMessage("编辑模式：点击并拖动警戒区域顶点进行编辑")
        else:
            # 退出编辑模式
            self.edit_area_button.setText("编辑警戒区域")
            
            # 停止预览
            self._stop_camera()
            
            self.statusBar().showMessage("编辑完成")
            
        # 重置选中状态
        self.selected_point = -1
    
    def _start_preview(self):
        """启动摄像头预览模式（不进行检测）"""
        # 初始化摄像头
        if self.current_camera == "video_file":
            video_path = self.video_path_label.text()
            if video_path == "选择视频文件..." or not os.path.exists(video_path):
                QMessageBox.warning(self, "警告", "请先选择有效的视频文件")
                self.edit_area_button.setChecked(False)
                self.edit_mode = False
                return
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(self.current_camera)
            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['video']['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['video']['height'])
        
        # 检查摄像头是否成功打开
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开视频源")
            self.edit_area_button.setChecked(False)
            self.edit_mode = False
            return
            
        # 更新UI状态
        self.preview_mode = True
        self.start_stop_button.setEnabled(False)  # 禁用监控按钮
        
        # 启动定时器
        self.timer.start(30)  # 30ms刷新一次，约等于33FPS
    
    def _update_threshold(self, value):
        """更新警报触发阈值"""
        self.warning_threshold = value
        self.statusBar().showMessage(f"警报触发阈值已更新为: {value} 秒")
    
    def _save_config(self):
        """保存警戒区域设置到配置文件"""
        try:
            self.config['fence']['area'] = self.fence_area
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                
            QMessageBox.information(self, "成功", "警戒区域设置已保存")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存配置失败: {e}")
    
    def _on_video_click(self, event: QMouseEvent):
        """处理视频区域鼠标点击事件"""
        if not self.edit_mode or self.original_frame is None:
            return
        
        # 计算点击位置在实际视频帧中的坐标
        label_size = self.video_label.size()
        frame_size = self.original_frame.shape[:2][::-1]  # (w, h)
        
        # 计算视频在标签中的实际显示尺寸
        scale_factor = min(label_size.width() / frame_size[0], 
                           label_size.height() / frame_size[1])
        
        video_width = int(frame_size[0] * scale_factor)
        video_height = int(frame_size[1] * scale_factor)
        
        # 计算视频在标签中的偏移量
        x_offset = (label_size.width() - video_width) // 2
        y_offset = (label_size.height() - video_height) // 2
        
        # 将点击位置从标签坐标转换为视频帧坐标
        if (event.x() >= x_offset and event.x() < x_offset + video_width and
            event.y() >= y_offset and event.y() < y_offset + video_height):
            
            # 转换到视频帧坐标系
            frame_x = int((event.x() - x_offset) / scale_factor)
            frame_y = int((event.y() - y_offset) / scale_factor)
            
            if event.button() == Qt.LeftButton:
                # 查找最近的点
                self.selected_point = find_nearest_point(
                    (frame_x, frame_y), self.fence_area, 
                    self.original_frame.shape, 20)
                
                if self.selected_point >= 0:
                    self.statusBar().showMessage(f"已选中顶点 {self.selected_point + 1}")
            
        # 更新显示
        if self.original_frame is not None:
            self._update_edit_view()
    
    def _start_monitoring(self):
        """开始监控"""
        # 初始化摄像头
        if self.current_camera == "video_file":
            video_path = self.video_path_label.text()
            if video_path == "选择视频文件..." or not os.path.exists(video_path):
                QMessageBox.warning(self, "警告", "请先选择有效的视频文件")
                return
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(self.current_camera)
            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['video']['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['video']['height'])
        
        # 检查摄像头是否成功打开
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开视频源")
            return
        
        # 加载模型
        try:
            weights_path = self.config['model']['weights']
            if not os.path.exists(weights_path):
                QMessageBox.critical(self, "错误", f"模型文件不存在: {weights_path}")
                return
                
            self.model = YOLO(weights_path).to(self.device)
            self.statusBar().showMessage(f"模型加载成功，运行于: {self.device}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型初始化错误: {e}")
            return
        
        # 更新UI状态
        self.is_running = True
        self.preview_mode = False
        self.start_stop_button.setText("停止监控")
        self.edit_area_button.setEnabled(False)  # 禁用编辑按钮
        
        # 启动定时器
        self.timer.start(10)  # 10ms刷新一次，约等于100FPS
    
    def _stop_monitoring(self):
        """停止监控"""
        self._stop_camera()
        
        # 更新UI状态
        self.is_running = False
        self.start_stop_button.setText("开始监控")
        self.video_label.setText("点击「编辑警戒区域」按钮开始预览和编辑")
        self.statusBar().showMessage("监控已停止")
        self.edit_area_button.setEnabled(True)  # 启用编辑按钮
        
        # 清除模型
        self.model = None
    
    def _stop_camera(self):
        """停止摄像头（用于监控和预览模式）"""
        # 停止定时器
        self.timer.stop()
        
        # 释放资源
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # 更新UI状态
        self.is_running = False
        self.preview_mode = False
        
        # 如果不在编辑模式，还原开始监控按钮
        if not self.edit_mode:
            self.start_stop_button.setText("开始监控")
            self.video_label.setText("点击「编辑警戒区域」按钮开始预览和编辑")
        
        self.start_stop_button.setEnabled(True)  # 启用监控按钮
        
        # 清除状态
        self.original_frame = None
        self.warning_active = False
        self.intrusion_start_time = None
    
    def _toggle_detection_box(self, state):
        """切换是否显示检测框"""
        self.show_detections = (state == Qt.Checked)
    
    def _toggle_fps(self, state):
        """切换是否显示FPS"""
        self.show_fps = (state == Qt.Checked)
    
    def _update_edit_view(self):
        """更新编辑模式视图"""
        if self.original_frame is None:
            return
            
        # 复制原始帧
        frame = self.original_frame.copy()
        
        # 绘制警戒区域
        frame = draw_fence_area(
            frame, self.fence_area, self.line_color, 
            self.line_thickness, (0, 255, 0, 64))
        
        # 将帧转换为Qt图像并显示
        self._display_frame(frame)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """处理鼠标移动事件"""
        if not self.edit_mode or self.selected_point < 0 or self.original_frame is None:
            return super().mouseMoveEvent(event)
            
        # 计算鼠标在视频帧中的位置
        label_pos = self.video_label.mapFromGlobal(event.globalPos())
        label_size = self.video_label.size()
        frame_size = self.original_frame.shape[:2][::-1]  # (w, h)
        
        # 计算视频在标签中的实际显示尺寸
        scale_factor = min(label_size.width() / frame_size[0], 
                          label_size.height() / frame_size[1])
        
        video_width = int(frame_size[0] * scale_factor)
        video_height = int(frame_size[1] * scale_factor)
        
        # 计算视频在标签中的偏移量
        x_offset = (label_size.width() - video_width) // 2
        y_offset = (label_size.height() - video_height) // 2
        
        # 将鼠标位置从标签坐标转换为视频帧坐标
        if (label_pos.x() >= x_offset and label_pos.x() < x_offset + video_width and
            label_pos.y() >= y_offset and label_pos.y() < y_offset + video_height):
            
            # 转换到视频帧坐标系，并归一化
            frame_x = (label_pos.x() - x_offset) / scale_factor
            frame_y = (label_pos.y() - y_offset) / scale_factor
            
            # 更新选中顶点的位置
            self.fence_area[self.selected_point][0] = frame_x / frame_size[0]
            self.fence_area[self.selected_point][1] = frame_y / frame_size[1]
            
            # 约束在0-1范围内
            self.fence_area[self.selected_point][0] = max(0, min(1, self.fence_area[self.selected_point][0]))
            self.fence_area[self.selected_point][1] = max(0, min(1, self.fence_area[self.selected_point][1]))
            
            # 更新显示
            self._update_edit_view()
            
        return super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event: QMouseEvent):
        """处理鼠标释放事件"""
        if self.edit_mode and self.selected_point >= 0:
            self.selected_point = -1
            self.statusBar().showMessage("顶点移动完成")
        
        return super().mouseReleaseEvent(event)
    
    def _update_frame(self):
        """更新视频帧"""
        if not self.cap:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            # 视频结束
            if self.current_camera == "video_file":
                # 如果是视频文件，循环播放
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    self._stop_camera()
                    return
            else:
                self._stop_camera()
                return
        
        # 保存原始帧（用于编辑模式）
        self.original_frame = frame.copy()
        
        # 计算FPS
        current_time = time.time()
        frame_time = current_time - self.prev_time
        self.fps = 1 / frame_time if frame_time > 0 else 0
        self.prev_time = current_time
        
        # 记录处理时间
        self.processing_times.append(frame_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        # 如果只是预览模式（编辑警戒区域）
        if self.preview_mode:
            # 绘制警戒区域
            frame = draw_fence_area(
                frame, self.fence_area, self.line_color, self.line_thickness, 
                (0, 255, 0, 64))  # 半透明的绿色填充
            
            # 在预览模式下显示提示信息
            info_text = "编辑模式：点击并拖动顶点调整警戒区域"
            frame = put_chinese_text(frame, info_text, (10, 30), (0, 255, 0), 20)
            
            # 显示处理后的帧
            self._display_frame(frame)
            return
        
        # 以下是监控模式的处理
        # 使用YOLO检测人
        start_detect = time.time()
        results = self.model(frame, conf=self.config['model']['confidence'], 
                            classes=self.config['model']['classes'])
        detect_time = time.time() - start_detect
        
        # 绘制警戒区域
        frame = draw_fence_area(
            frame, self.fence_area, self.line_color, self.line_thickness)
        
        # 准备检测框列表用于跟踪
        detections = []
        
        # 处理检测结果
        if len(results) > 0:
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box[:4])
                    detections.append([x1, y1, x2, y2])
        
        # 使用ByteTracker更新跟踪
        tracks = self.tracker.update(detections)
        
        # 当前帧没有入侵者
        intrusion_detected = False
        current_track_ids = set()
        
        # 处理每个跟踪目标
        for track in tracks:
            track_id, x1, y1, x2, y2 = map(int, track)
            current_track_ids.add(track_id)
            
            # 检查是否在警戒区域内
            in_area, judgment_point = is_bbox_in_area([x1, y1, x2, y2], self.fence_area, frame.shape)
            
            # 显示判断点
            if self.show_judgment_point:
                frame = draw_judgment_point(frame, judgment_point, in_area)
            
            # 显示检测框和ID
            if self.show_detections:
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制ID标签
                id_text = f"ID: {track_id}"
                cv2.putText(frame, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 处理入侵逻辑
            if in_area:
                # 如果是新的跟踪目标或现有目标但未标记入侵
                if track_id not in self.track_status or not self.track_status[track_id]['intrusion_start']:
                    self.track_status[track_id] = {
                        'intrusion_start': time.time(),
                        'warning': False
                    }
                
                # 计算入侵持续时间
                intrusion_duration = time.time() - self.track_status[track_id]['intrusion_start']
                
                # 显示持续时间
                duration_text = f"ID:{track_id} 入侵持续: {intrusion_duration:.1f}秒"
                y_pos = 110 + (track_id % 10) * 30  # 避免文本重叠
                frame = put_chinese_text(frame, duration_text, (10, y_pos), (0, 0, 255), 20)
                
                # 如果超过阈值，触发警报
                if intrusion_duration >= self.warning_threshold:
                    self.track_status[track_id]['warning'] = True
                
                # 标记当前帧有入侵
                intrusion_detected = True
                
                self.statusBar().showMessage(f"检测到ID:{track_id}入侵！持续时间: {intrusion_duration:.1f}秒")
            else:
                # 如果不在警戒区域，重置该ID的入侵状态
                if track_id in self.track_status:
                    self.track_status[track_id]['intrusion_start'] = None
                    # 保持警报状态不变，直到完全消失
        
        # 清理不再出现的轨迹
        track_ids_to_remove = []
        for track_id in self.track_status:
            if track_id not in current_track_ids:
                track_ids_to_remove.append(track_id)
        
        for track_id in track_ids_to_remove:
            del self.track_status[track_id]
        
        # 检查是否有任何ID的警报被激活
        self.warning_active = any(status.get('warning', False) for status in self.track_status.values())
        
        # 显示FPS和处理时间信息
        if self.show_fps:
            fps_text = f"FPS: {int(self.fps)}"
            detect_text = f"检测时间: {detect_time*1000:.1f}ms"
            
            # 使用PIL渲染中文
            frame = put_chinese_text(frame, fps_text, (10, 30), (0, 255, 0), 30)
            frame = put_chinese_text(frame, detect_text, (10, 70), (0, 255, 0), 20)
        
        # 如果有警报激活，显示警告
        if self.warning_active:
            frame = draw_warning(frame, self.warning_text, self.warning_color)
            
            # 更新状态栏
            self.statusBar().showMessage("警告！检测到持续越界")
        elif not intrusion_detected:
            # 更新状态栏 - 显示性能信息
            avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
            avg_fps = 1 / avg_time if avg_time > 0 else 0
            self.statusBar().showMessage(f"正常监控中 | 检测时间: {detect_time*1000:.1f}ms | 平均FPS: {avg_fps:.1f}")
        
        # 显示处理后的帧
        self._display_frame(frame)
    
    def _display_frame(self, frame):
        """将OpenCV帧显示在UI上"""
        # 将OpenCV图像转换为Qt图像
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 调整大小以适应标签
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
        # 显示在UI上
        self.video_label.setPixmap(scaled_pixmap)
    
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # 停止监控
        if self.is_running or self.preview_mode:
            self._stop_camera()
        event.accept()
    
if __name__ == "__main__":
    # 允许使用高DPI缩放
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    
    # 设置全局中文字体
    font = QFont("SimHei", 9)
    app.setFont(font)
    
    # 创建窗口
    window = ElectronicFenceGUI()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_()) 