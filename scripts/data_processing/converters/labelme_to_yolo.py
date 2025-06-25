"""
LabelMe JSON格式转YOLO TXT格式转换器
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
from .base_converter import BaseConverter

class LabelmeToYoloConverter(BaseConverter):
    """LabelMe格式转YOLO格式转换器"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def validate_input(self, input_path: str) -> bool:
        """验证LabelMe JSON文件是否存在且格式正确"""
        if not os.path.exists(input_path):
            self.logger.error(f"输入路径不存在: {input_path}")
            return False
        
        # 检查是否包含JSON文件
        json_files = list(Path(input_path).glob("*.json"))
        if not json_files:
            self.logger.error(f"未找到JSON文件: {input_path}")
            return False
        
        # 验证至少一个JSON文件的格式
        try:
            sample_json = json_files[0]
            with open(sample_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查必要的字段
            required_fields = ['imageHeight', 'imageWidth', 'shapes']
            for field in required_fields:
                if field not in data:
                    self.logger.error(f"JSON文件缺少必要字段: {field}")
                    return False
            
            self.logger.info(f"找到 {len(json_files)} 个JSON文件")
            return True
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析错误: {e}")
            return False
        except Exception as e:
            self.logger.error(f"验证输入时发生错误: {e}")
            return False
    
    def _parse_labelme_file(self, json_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        解析单个LabelMe JSON文件
        
        Returns:
            (图像信息, 标注列表)
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图像信息
        image_info = {
            'filename': data.get('imagePath', json_path.stem + '.jpg'),
            'width': data['imageWidth'],
            'height': data['imageHeight']
        }
        
        # 获取标注信息
        annotations = []
        for shape in data.get('shapes', []):
            if shape['shape_type'] not in ['rectangle', 'polygon']:
                self.logger.warning(f"跳过不支持的形状类型: {shape['shape_type']}")
                continue
            
            label = shape['label']
            points = shape['points']
            
            annotation = {
                'label': label,
                'shape_type': shape['shape_type'],
                'points': points,
                'group_id': shape.get('group_id'),
                'flags': shape.get('flags', {})
            }
            
            annotations.append(annotation)
        
        return image_info, annotations
    
    def _polygon_to_bbox(self, points: List[List[float]]) -> List[float]:
        """将多边形转换为边界框"""
        points = np.array(points)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        xmin = float(np.min(x_coords))
        ymin = float(np.min(y_coords))
        xmax = float(np.max(x_coords))
        ymax = float(np.max(y_coords))
        
        return [xmin, ymin, xmax, ymax]
    
    def _rectangle_to_bbox(self, points: List[List[float]]) -> List[float]:
        """将矩形点转换为边界框"""
        if len(points) != 2:
            raise ValueError("矩形应该有2个点")
        
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        xmin = min(x1, x2)
        ymin = min(y1, y2)
        xmax = max(x1, x2)
        ymax = max(y1, y2)
        
        return [xmin, ymin, xmax, ymax]
    
    def _convert_bbox_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """将边界框转换为YOLO格式"""
        xmin, ymin, xmax, ymax = bbox
        
        # 计算中心点和宽高
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        
        # 归一化
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return [x_center, y_center, width, height]
    
    def _polygon_to_yolo_segmentation(self, points: List[List[float]], img_width: int, img_height: int) -> List[float]:
        """将多边形转换为YOLO分割格式（归一化的点坐标序列）"""
        normalized_points = []
        for point in points:
            x, y = point
            normalized_points.extend([x / img_width, y / img_height])
        return normalized_points
    
    def convert(self, 
                input_path: str, 
                output_path: str, 
                class_mapping: Optional[Dict[str, int]] = None,
                output_format: str = 'detection') -> Dict[str, Any]:
        """
        执行LabelMe到YOLO的转换
        
        Args:
            input_path: LabelMe数据集路径
            output_path: YOLO输出路径
            class_mapping: 可选的类别映射
            output_format: 输出格式 ('detection' 或 'segmentation')
            
        Returns:
            转换结果信息
        """
        if not self.validate_input(input_path):
            raise ValueError(f"输入验证失败: {input_path}")
        
        if output_format not in ['detection', 'segmentation']:
            raise ValueError("output_format 必须是 'detection' 或 'segmentation'")
        
        # 创建输出目录
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 收集所有类别名称
        all_classes = set()
        json_files = list(Path(input_path).glob("*.json"))
        
        self.logger.info(f"第一遍扫描: 收集类别信息...")
        for json_file in json_files:
            try:
                _, annotations = self._parse_labelme_file(json_file)
                for ann in annotations:
                    all_classes.add(ann['label'])
            except Exception as e:
                self.logger.warning(f"跳过文件 {json_file}: {e}")
                continue
        
        # 创建类别映射
        if class_mapping:
            category_mapping = class_mapping
        else:
            category_mapping = {class_name: idx for idx, class_name in enumerate(sorted(all_classes))}
        
        self.class_names = list(category_mapping.keys())
        self.logger.info(f"类别映射: {category_mapping}")
        
        # 转换标注
        converted_files = []
        converted_annotations = 0
        skipped_annotations = 0
        
        self.logger.info(f"开始转换 {len(json_files)} 个JSON文件...")
        
        for i, json_file in enumerate(json_files, 1):
            if i % 50 == 0:
                self.logger.info(f"进度: [{i}/{len(json_files)}]")
            
            try:
                image_info, annotations = self._parse_labelme_file(json_file)
                
                # 创建对应的txt文件
                txt_filename = json_file.stem + '.txt'
                txt_path = output_path / txt_filename
                
                yolo_annotations = []
                
                for ann in annotations:
                    label = ann['label']
                    if label not in category_mapping:
                        self.logger.warning(f"未知类别: {label}")
                        skipped_annotations += 1
                        continue
                    
                    class_id = category_mapping[label]
                    points = ann['points']
                    shape_type = ann['shape_type']
                    
                    try:
                        if output_format == 'detection':
                            # 检测格式：转换为边界框
                            if shape_type == 'rectangle':
                                bbox = self._rectangle_to_bbox(points)
                            elif shape_type == 'polygon':
                                bbox = self._polygon_to_bbox(points)
                            else:
                                skipped_annotations += 1
                                continue
                            
                            yolo_bbox = self._convert_bbox_to_yolo(
                                bbox, image_info['width'], image_info['height']
                            )
                            yolo_annotations.append(f"{class_id} {' '.join(map(str, yolo_bbox))}")
                            
                        elif output_format == 'segmentation':
                            # 分割格式：保留多边形点
                            if shape_type == 'polygon':
                                seg_points = self._polygon_to_yolo_segmentation(
                                    points, image_info['width'], image_info['height']
                                )
                                yolo_annotations.append(f"{class_id} {' '.join(map(str, seg_points))}")
                            elif shape_type == 'rectangle':
                                # 将矩形转换为多边形
                                xmin, ymin, xmax, ymax = self._rectangle_to_bbox(points)
                                rect_points = [
                                    [xmin, ymin], [xmax, ymin], 
                                    [xmax, ymax], [xmin, ymax]
                                ]
                                seg_points = self._polygon_to_yolo_segmentation(
                                    rect_points, image_info['width'], image_info['height']
                                )
                                yolo_annotations.append(f"{class_id} {' '.join(map(str, seg_points))}")
                            else:
                                skipped_annotations += 1
                                continue
                        
                        converted_annotations += 1
                        
                    except Exception as e:
                        self.logger.warning(f"转换标注失败: {e}")
                        skipped_annotations += 1
                        continue
                
                # 写入YOLO格式文件
                if yolo_annotations:
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(yolo_annotations) + '\n')
                    converted_files.append(txt_filename)
                else:
                    # 创建空文件（如果图像没有标注）
                    txt_path.touch()
                    converted_files.append(txt_filename)
                
            except Exception as e:
                self.logger.error(f"转换文件失败 {json_file}: {e}")
                continue
        
        # 保存类别名称文件
        classes_file = output_path / 'classes.txt'
        with open(classes_file, 'w', encoding='utf-8') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        
        result = {
            'converted_files': converted_files,
            'class_names': self.class_names,
            'total_json_files': len(json_files),
            'converted_label_files': len(converted_files),
            'converted_annotations': converted_annotations,
            'skipped_annotations': skipped_annotations,
            'class_mapping': category_mapping,
            'output_format': output_format
        }
        
        self.logger.info(f"转换完成:")
        self.logger.info(f"  - JSON文件数: {result['total_json_files']}")
        self.logger.info(f"  - 生成标签文件: {result['converted_label_files']}")
        self.logger.info(f"  - 转换标注数: {result['converted_annotations']}")
        self.logger.info(f"  - 跳过标注数: {result['skipped_annotations']}")
        self.logger.info(f"  - 类别数: {len(self.class_names)}")
        self.logger.info(f"  - 输出格式: {output_format}")
        
        return result
