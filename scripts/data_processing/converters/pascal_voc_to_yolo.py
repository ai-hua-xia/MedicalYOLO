"""
Pascal VOC XML格式转YOLO TXT格式转换器
"""
import xml.etree.ElementTree as ET
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from .base_converter import BaseConverter

class PascalVocToYoloConverter(BaseConverter):
    """Pascal VOC格式转YOLO格式转换器"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def validate_input(self, input_path: str) -> bool:
        """验证Pascal VOC XML文件是否存在且格式正确"""
        if not os.path.exists(input_path):
            self.logger.error(f"输入路径不存在: {input_path}")
            return False
        
        # 检查是否包含XML文件
        xml_files = list(Path(input_path).glob("*.xml"))
        if not xml_files:
            self.logger.error(f"未找到XML文件: {input_path}")
            return False
        
        # 验证至少一个XML文件的格式
        try:
            sample_xml = xml_files[0]
            tree = ET.parse(sample_xml)
            root = tree.getroot()
            
            # 检查必要的元素
            if root.find('size') is None:
                self.logger.error("XML文件缺少size元素")
                return False
                
            if not root.findall('object'):
                self.logger.warning("XML文件中没有object元素")
            
            self.logger.info(f"找到 {len(xml_files)} 个XML文件")
            return True
            
        except ET.ParseError as e:
            self.logger.error(f"XML解析错误: {e}")
            return False
        except Exception as e:
            self.logger.error(f"验证输入时发生错误: {e}")
            return False
    
    def _parse_xml_file(self, xml_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        解析单个XML文件
        
        Returns:
            (图像信息, 标注列表)
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取图像信息
        size_elem = root.find('size')
        if size_elem is None:
            raise ValueError(f"XML文件 {xml_path} 缺少size信息")
        
        width = int(size_elem.find('width').text)
        height = int(size_elem.find('height').text)
        depth = int(size_elem.find('depth').text) if size_elem.find('depth') is not None else 3
        
        filename_elem = root.find('filename')
        filename = filename_elem.text if filename_elem is not None else xml_path.stem + '.jpg'
        
        image_info = {
            'filename': filename,
            'width': width,
            'height': height,
            'depth': depth
        }
        
        # 获取标注信息
        annotations = []
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is None:
                continue
            
            class_name = name_elem.text
            
            # 获取边界框
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue
            
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # 获取其他属性
            difficult = obj.find('difficult')
            difficult = int(difficult.text) if difficult is not None else 0
            
            truncated = obj.find('truncated')
            truncated = int(truncated.text) if truncated is not None else 0
            
            annotations.append({
                'class_name': class_name,
                'bbox': [xmin, ymin, xmax, ymax],
                'difficult': difficult,
                'truncated': truncated
            })
        
        return image_info, annotations
    
    def _convert_bbox_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """将Pascal VOC bbox转换为YOLO格式"""
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
    
    def convert(self, 
                input_path: str, 
                output_path: str, 
                class_mapping: Optional[Dict[str, int]] = None,
                include_difficult: bool = False) -> Dict[str, Any]:
        """
        执行Pascal VOC到YOLO的转换
        
        Args:
            input_path: Pascal VOC数据集路径
            output_path: YOLO输出路径
            class_mapping: 可选的类别映射
            include_difficult: 是否包含difficult标记的对象
            
        Returns:
            转换结果信息
        """
        if not self.validate_input(input_path):
            raise ValueError(f"输入验证失败: {input_path}")
        
        # 创建输出目录
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 收集所有类别名称
        all_classes = set()
        xml_files = list(Path(input_path).glob("*.xml"))
        
        self.logger.info(f"第一遍扫描: 收集类别信息...")
        for xml_file in xml_files:
            try:
                _, annotations = self._parse_xml_file(xml_file)
                for ann in annotations:
                    all_classes.add(ann['class_name'])
            except Exception as e:
                self.logger.warning(f"跳过文件 {xml_file}: {e}")
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
        
        self.logger.info(f"开始转换 {len(xml_files)} 个XML文件...")
        
        for i, xml_file in enumerate(xml_files, 1):
            if i % 50 == 0:
                self.logger.info(f"进度: [{i}/{len(xml_files)}]")
            
            try:
                image_info, annotations = self._parse_xml_file(xml_file)
                
                # 创建对应的txt文件
                txt_filename = xml_file.stem + '.txt'
                txt_path = output_path / txt_filename
                
                yolo_annotations = []
                
                for ann in annotations:
                    # 跳过difficult对象（如果设置）
                    if not include_difficult and ann['difficult']:
                        skipped_annotations += 1
                        continue
                    
                    class_name = ann['class_name']
                    if class_name not in category_mapping:
                        self.logger.warning(f"未知类别: {class_name}")
                        continue
                    
                    class_id = category_mapping[class_name]
                    
                    # 转换边界框
                    yolo_bbox = self._convert_bbox_to_yolo(
                        ann['bbox'], 
                        image_info['width'], 
                        image_info['height']
                    )
                    
                    yolo_annotations.append(f"{class_id} {' '.join(map(str, yolo_bbox))}")
                    converted_annotations += 1
                
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
                self.logger.error(f"转换文件失败 {xml_file}: {e}")
                continue
        
        # 保存类别名称文件
        classes_file = output_path / 'classes.txt'
        with open(classes_file, 'w', encoding='utf-8') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        
        result = {
            'converted_files': converted_files,
            'class_names': self.class_names,
            'total_xml_files': len(xml_files),
            'converted_label_files': len(converted_files),
            'converted_annotations': converted_annotations,
            'skipped_annotations': skipped_annotations,
            'class_mapping': category_mapping
        }
        
        self.logger.info(f"转换完成:")
        self.logger.info(f"  - XML文件数: {result['total_xml_files']}")
        self.logger.info(f"  - 生成标签文件: {result['converted_label_files']}")
        self.logger.info(f"  - 转换标注数: {result['converted_annotations']}")
        self.logger.info(f"  - 跳过标注数: {result['skipped_annotations']}")
        self.logger.info(f"  - 类别数: {len(self.class_names)}")
        
        return result
