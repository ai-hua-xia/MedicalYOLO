import os
import json
import xml.etree.ElementTree as ET
import yaml
import shutil
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional, Tuple
import random

class BaseConverter:
    """基础转换器类"""
    
    def extract_classes(self, data_path: str) -> List[str]:
        """提取数据集中的所有类别"""
        raise NotImplementedError
    
    def convert_to_yolo(self, data_path: str, output_dir: str, class_names: List[str]) -> None:
        """转换为YOLO格式"""
        raise NotImplementedError

class COCOConverter(BaseConverter):
    """COCO格式转换器"""
    
    def extract_classes(self, json_file: str) -> List[str]:
        """从COCO JSON文件中提取所有类别"""
        with open(json_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        categories = coco_data.get('categories', [])
        class_names = [cat['name'] for cat in categories]
        return sorted(class_names)
    
    def convert_to_yolo(self, json_file: str, output_dir: str, class_names: List[str]) -> None:
        """将COCO JSON转换为YOLO格式"""
        with open(json_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 创建映射
        images = {img['id']: img for img in coco_data['images']}
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # 按图像分组标注
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        os.makedirs(output_dir, exist_ok=True)
        
        for image_id, annotations in annotations_by_image.items():
            if image_id not in images:
                continue
                
            image_info = images[image_id]
            img_width = image_info['width']
            img_height = image_info['height']
            
            yolo_annotations = []
            
            for ann in annotations:
                category_name = categories.get(ann['category_id'])
                if not category_name or category_name not in class_names:
                    continue
                
                class_id = class_names.index(category_name)
                
                # COCO bbox格式: [x, y, width, height]
                x, y, width, height = ann['bbox']
                
                # 转换为YOLO格式
                x_center = (x + width / 2.0) / img_width
                y_center = (y + height / 2.0) / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
            
            # 保存YOLO格式标注文件
            txt_filename = os.path.splitext(image_info['file_name'])[0] + '.txt'
            txt_path = os.path.join(output_dir, txt_filename)
            
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

class PascalVOCConverter(BaseConverter):
    """Pascal VOC格式转换器"""
    
    def extract_classes(self, xml_dir: str) -> List[str]:
        """从Pascal VOC XML文件中提取所有类别"""
        class_names = set()
        xml_files = Path(xml_dir).glob('*.xml')
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name:
                        class_names.add(class_name)
            except Exception as e:
                print(f"解析XML文件失败 {xml_file}: {e}")
        
        return sorted(list(class_names))
    
    def convert_to_yolo(self, xml_dir: str, img_dir: str, output_dir: str, class_names: List[str]) -> None:
        """将Pascal VOC转换为YOLO格式"""
        os.makedirs(output_dir, exist_ok=True)
        xml_files = list(Path(xml_dir).glob('*.xml'))
        
        for xml_file in xml_files:
            try:
                # 查找对应的图像文件
                img_name = xml_file.stem
                img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                img_path = None
                
                for ext in img_extensions:
                    potential_path = Path(img_dir) / (img_name + ext)
                    if potential_path.exists():
                        img_path = potential_path
                        break
                
                if not img_path:
                    print(f"找不到对应的图像文件: {img_name}")
                    continue
                
                # 获取图像尺寸
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                
                # 解析XML
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                yolo_annotations = []
                
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in class_names:
                        continue
                        
                    class_id = class_names.index(class_name)
                    
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # 转换为YOLO格式
                    x_center = (xmin + xmax) / 2.0 / img_width
                    y_center = (ymin + ymax) / 2.0 / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    
                    yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # 保存YOLO格式标注
                txt_path = Path(output_dir) / (img_name + '.txt')
                
                with open(txt_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
            except Exception as e:
                print(f"转换失败 {xml_file}: {e}")

class DataConverter:
    """统一的数据转换入口"""
    
    def __init__(self):
        self.converters = {
            'coco': COCOConverter(),
            'pascal_voc': PascalVOCConverter()
        }
    
    def convert(self, input_dir: str, output_dir: str, format_type: str, 
                class_names: Optional[List[str]] = None, img_dir: Optional[str] = None) -> List[str]:
        """
        统一转换接口
        
        Args:
            input_dir: 输入标注文件目录
            output_dir: 输出目录
            format_type: 原始标注格式类型 ('coco' 或 'pascal_voc')
            class_names: 指定的类别列表，如果为None则自动提取
            img_dir: 图像文件目录（Pascal VOC格式需要）
        
        Returns:
            实际使用的类别列表
        """
        if format_type not in self.converters:
            raise ValueError(f"不支持的格式类型: {format_type}")
        
        converter = self.converters[format_type]
        
        # 自动提取类别或使用指定类别
        if class_names is None:
            if format_type == 'coco':
                json_files = list(Path(input_dir).glob('*.json'))
                if not json_files:
                    raise FileNotFoundError(f"在 {input_dir} 中未找到JSON文件")
                class_names = converter.extract_classes(str(json_files[0]))
            elif format_type == 'pascal_voc':
                class_names = converter.extract_classes(input_dir)
        
        print(f"检测到类别: {class_names}")
        
        # 执行转换
        if format_type == 'coco':
            json_files = list(Path(input_dir).glob('*.json'))
            for json_file in json_files:
                print(f"转换文件: {json_file}")
                converter.convert_to_yolo(str(json_file), output_dir, class_names)
        elif format_type == 'pascal_voc':
            if not img_dir:
                raise ValueError("Pascal VOC格式需要指定图像目录")
            converter.convert_to_yolo(input_dir, img_dir, output_dir, class_names)
        
        print(f"转换完成，输出目录: {output_dir}")
        return class_names

class DatasetSplitter:
    """数据集划分器"""
    
    def __init__(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("训练、验证、测试集比例之和必须为1")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def split_dataset(self, labels_dir: str, images_dir: str, output_dir: str, class_names: List[str]) -> str:
        """
        划分数据集并生成data.yaml配置文件
        
        Args:
            labels_dir: YOLO格式标签目录
            images_dir: 图像文件目录
            output_dir: 输出数据集目录
            class_names: 类别名称列表
        
        Returns:
            生成的data.yaml文件路径
        """
        # 获取所有标签文件
        label_files = list(Path(labels_dir).glob('*.txt'))
        
        if not label_files:
            raise FileNotFoundError(f"在 {labels_dir} 中未找到标签文件")
        
        # 随机打乱
        random.shuffle(label_files)
        
        # 计算划分点
        total_files = len(label_files)
        train_split = int(total_files * self.train_ratio)
        val_split = int(total_files * (self.train_ratio + self.val_ratio))
        
        # 划分文件列表
        train_files = label_files[:train_split]
        val_files = label_files[train_split:val_split]
        test_files = label_files[val_split:]
        
        print(f"数据集划分: 训练集={len(train_files)}, 验证集={len(val_files)}, 测试集={len(test_files)}")
        
        # 创建输出目录结构
        splits = {'train': train_files, 'val': val_files, 'test': test_files}
        
        for split_name, files in splits.items():
            # 创建目录
            split_img_dir = Path(output_dir) / split_name / 'images'
            split_label_dir = Path(output_dir) / split_name / 'labels'
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_label_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            for label_file in files:
                # 复制标签文件
                shutil.copy2(label_file, split_label_dir / label_file.name)
                
                # 查找并复制对应的图像文件
                img_name = label_file.stem
                img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                
                for ext in img_extensions:
                    img_file = Path(images_dir) / (img_name + ext)
                    if img_file.exists():
                        shutil.copy2(img_file, split_img_dir / img_file.name)
                        break
                else:
                    print(f"警告: 未找到对应的图像文件 {img_name}")
        
        # 生成data.yaml配置文件
        config_dir = Path('configs')
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / 'data.yaml'
        
        config_data = {
            'path': str(Path(output_dir).absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        
        print(f"配置文件已生成: {config_path}")
        return str(config_path)

def main():
    """主函数示例"""
    try:
        # 第一层：格式转换
        converter = DataConverter()
        
        # 使用现有的raw目录结构
        class_names = converter.convert(
            input_dir='data/raw/annotations',
            output_dir='data/temp_labels',
            format_type='coco'
        )
        
        # 第二层：数据集划分
        splitter = DatasetSplitter(train_ratio=0.7, val_ratio=0.1, test_ratio=0.1)
        
        config_path = splitter.split_dataset(
            labels_dir='data/temp_labels',
            images_dir='data/raw',
            output_dir='data',
            class_names=class_names
        )
        
        print(f"数据处理完成，配置文件: {config_path}")
        
    finally:
        # 清理临时文件
        temp_dir = Path('data/temp_labels')
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("临时文件已清理")

if __name__ == "__main__":
    main()
