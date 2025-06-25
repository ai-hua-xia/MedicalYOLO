"""
数据集分割工具
支持按比例或按数量分割数据集
"""
import os
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
import json
from collections import defaultdict

class DatasetSplitter:
    """数据集分割器"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.logger = logging.getLogger(__name__)
        random.seed(seed)
    
    def split_by_ratio(self,
                      data_dir: str,
                      output_dir: str,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      labels_dir: str = "data/labels",  # 新增参数
                      image_extensions: List[str] = None) -> Dict[str, int]:
        """
        按比例分割数据集
        
        Args:
            data_dir: 数据目录路径
            output_dir: 输出目录路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            image_extensions: 图像文件扩展名列表
            
        Returns:
            分割结果统计
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # 验证比例
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"比例总和必须为1.0，当前为: {total_ratio}")
        
        # 获取所有图像文件
        data_path = Path(data_dir)
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(data_path.glob(f"*{ext}"))
            image_files.extend(data_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"在 {data_dir} 中未找到图像文件")
        
        # 随机打乱
        random.shuffle(image_files)
        
        total_count = len(image_files)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        test_count = total_count - train_count - val_count
        
        # 分割文件列表
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]
        
        # 创建输出目录结构
        output_path = Path(output_dir)
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        results = {}
        
        labels_root = Path(labels_dir)
        
        for split_name, files in splits.items():
            if not files:
                continue
                
            split_dir = output_path / split_name
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            copied_images = 0
            copied_labels = 0
            
            for img_file in files:
                # 复制图像文件
                dst_img = images_dir / img_file.name
                shutil.copy2(img_file, dst_img)
                copied_images += 1
                
                # 查找对应的标签文件
                label_file = labels_root / (img_file.stem + '.txt')
                if label_file.exists():
                    dst_label = labels_dir / label_file.name
                    shutil.copy2(label_file, dst_label)
                    copied_labels += 1
            
            results[split_name] = {
                'images': copied_images,
                'labels': copied_labels
            }
            
            self.logger.info(f"{split_name} 集: {copied_images} 图像, {copied_labels} 标签")
        
        # 保存分割信息
        split_info = {
            'seed': self.seed,
            'ratios': {
                'train': train_ratio,
                'val': val_ratio,
                'test': test_ratio
            },
            'counts': {
                'total': total_count,
                'train': train_count,
                'val': val_count,
                'test': test_count
            },
            'results': results
        }
        
        info_file = output_path / 'split_info.json'
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        
        return results
    
    def split_by_class(self,
                      data_dir: str,
                      output_dir: str,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1) -> Dict[str, Dict[str, int]]:
        """
        按类别平衡分割数据集
        
        Args:
            data_dir: 数据目录路径
            output_dir: 输出目录路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            按类别的分割结果统计
        """
        # 解析标签文件，按类别分组
        data_path = Path(data_dir)
        labels_dir = data_path / 'labels' if (data_path / 'labels').exists() else data_path
        
        class_files = defaultdict(list)
        
        for label_file in labels_dir.glob('*.txt'):
            img_name = label_file.stem
            
            # 读取标签文件中的类别
            classes_in_file = set()
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            classes_in_file.add(class_id)
            except Exception as e:
                self.logger.warning(f"无法读取标签文件 {label_file}: {e}")
                continue
            
            # 将文件分配给每个包含的类别
            for class_id in classes_in_file:
                class_files[class_id].append(img_name)
        
        if not class_files:
            raise ValueError("未找到有效的标签文件")
        
        # 为每个类别进行分割
        output_path = Path(output_dir)
        all_train_files = set()
        all_val_files = set()
        all_test_files = set()
        
        class_results = {}
        
        for class_id, files in class_files.items():
            random.shuffle(files)
            
            total_count = len(files)
            train_count = int(total_count * train_ratio)
            val_count = int(total_count * val_ratio)
            
            train_files = set(files[:train_count])
            val_files = set(files[train_count:train_count + val_count])
            test_files = set(files[train_count + val_count:])
            
            all_train_files.update(train_files)
            all_val_files.update(val_files)
            all_test_files.update(test_files)
            
            class_results[class_id] = {
                'total': total_count,
                'train': len(train_files),
                'val': len(val_files),
                'test': len(test_files)
            }
        
        # 复制文件到对应目录
        self._copy_split_files(data_path, output_path, {
            'train': all_train_files,
            'val': all_val_files,
            'test': all_test_files
        })
        
        return class_results
    
    def _copy_split_files(self, 
                         source_dir: Path, 
                         output_dir: Path, 
                         file_splits: Dict[str, set]):
        """复制分割后的文件"""
        for split_name, file_names in file_splits.items():
            if not file_names:
                continue
            
            split_dir = output_dir / split_name
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            for file_name in file_names:
                # 查找图像文件
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img_file = source_dir / f"{file_name}{ext}"
                    if img_file.exists():
                        shutil.copy2(img_file, images_dir / img_file.name)
                        break
                
                # 复制标签文件
                label_file = source_dir / 'labels' / f"{file_name}.txt"
                if not label_file.exists():
                    label_file = source_dir / f"{file_name}.txt"
                
                if label_file.exists():
                    shutil.copy2(label_file, labels_dir / f"{file_name}.txt")
    
    def create_yolo_dataset_yaml(self,
                                output_dir: str,
                                class_names: List[str],
                                dataset_name: str = "custom_dataset") -> str:
        """
        创建YOLO数据集配置文件
        
        Args:
            output_dir: 输出目录
            class_names: 类别名称列表
            dataset_name: 数据集名称
            
        Returns:
            配置文件路径
        """
        output_path = Path(output_dir)
        
        yaml_content = f"""# {dataset_name} dataset configuration
# Generated by MedicalYOLO DatasetSplitter

path: {output_path.absolute()}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images      # val images (relative to 'path')
test: test/images    # test images (optional, relative to 'path')

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}  # class names
"""
        
        yaml_file = output_path / 'dataset.yaml'
        with open(yaml_file, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        self.logger.info(f"YOLO数据集配置文件已保存: {yaml_file}")
        return str(yaml_file)
