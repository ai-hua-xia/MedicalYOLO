"""
数据验证工具
用于验证数据集的完整性和正确性
"""
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import logging
from collections import defaultdict, Counter

def validate_yolo_dataset(dataset_path: str, 
                         check_images: bool = True,
                         image_extensions: List[str] = None) -> Dict[str, any]:
    """
    验证YOLO数据集的完整性
    
    Args:
        dataset_path: 数据集路径
        check_images: 是否检查对应的图像文件
        image_extensions: 支持的图像扩展名
        
    Returns:
        验证结果字典
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    logger = logging.getLogger(__name__)
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise ValueError(f"数据集路径不存在: {dataset_path}")
    
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {},
        'files': {
            'label_files': [],
            'image_files': [],
            'orphan_labels': [],
            'orphan_images': []
        }
    }
    
    # 查找标签文件
    label_files = list(dataset_path.glob("**/*.txt"))
    results['files']['label_files'] = [str(f) for f in label_files]
    
    if check_images:
        # 查找图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f"**/*{ext}"))
            image_files.extend(dataset_path.glob(f"**/*{ext.upper()}"))
        
        results['files']['image_files'] = [str(f) for f in image_files]
        
        # 创建文件名集合（不包含扩展名）
        label_stems = {f.stem for f in label_files}
        image_stems = {f.stem for f in image_files}
        
        # 查找孤立文件
        orphan_labels = label_stems - image_stems
        orphan_images = image_stems - label_stems
        
        results['files']['orphan_labels'] = list(orphan_labels)
        results['files']['orphan_images'] = list(orphan_images)
        
        if orphan_labels:
            results['warnings'].append(f"发现 {len(orphan_labels)} 个没有对应图像的标签文件")
        
        if orphan_images:
            results['warnings'].append(f"发现 {len(orphan_images)} 个没有对应标签的图像文件")
    
    # 验证标签文件格式
    class_ids = set()
    bbox_errors = []
    empty_files = []
    annotation_count = 0
    
    for label_file in label_files:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines or all(not line.strip() for line in lines):
                empty_files.append(str(label_file))
                continue
            
            for line_no, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    bbox_errors.append(f"{label_file}:{line_no} - 格式错误: {line}")
                    continue
                
                try:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    
                    class_ids.add(class_id)
                    annotation_count += 1
                    
                    # 验证bbox范围
                    x_center, y_center, width, height = bbox
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                           0 < width <= 1 and 0 < height <= 1):
                        bbox_errors.append(f"{label_file}:{line_no} - bbox超出范围: {bbox}")
                    
                except ValueError as e:
                    bbox_errors.append(f"{label_file}:{line_no} - 数值错误: {line}")
                
        except Exception as e:
            results['errors'].append(f"读取文件失败 {label_file}: {e}")
    
    # 统计信息
    results['statistics'] = {
        'total_label_files': len(label_files),
        'total_image_files': len(results['files']['image_files']) if check_images else 0,
        'empty_label_files': len(empty_files),
        'total_annotations': annotation_count,
        'unique_classes': len(class_ids),
        'class_ids': sorted(list(class_ids)),
        'bbox_errors': len(bbox_errors)
    }
    
    # 添加错误和警告
    if bbox_errors:
        results['errors'].extend(bbox_errors[:10])  # 只显示前10个错误
        if len(bbox_errors) > 10:
            results['errors'].append(f"... 还有 {len(bbox_errors) - 10} 个bbox错误")
    
    if empty_files:
        results['warnings'].append(f"发现 {len(empty_files)} 个空标签文件")
    
    if not class_ids:
        results['errors'].append("未找到任何有效的类别标注")
    
    # 检查类别ID连续性
    if class_ids:
        min_id, max_id = min(class_ids), max(class_ids)
        expected_ids = set(range(min_id, max_id + 1))
        missing_ids = expected_ids - class_ids
        if missing_ids:
            results['warnings'].append(f"类别ID不连续，缺失: {sorted(missing_ids)}")
    
    # 设置验证状态
    results['valid'] = len(results['errors']) == 0
    
    logger.info(f"数据集验证完成: {'通过' if results['valid'] else '失败'}")
    logger.info(f"标签文件: {results['statistics']['total_label_files']}")
    logger.info(f"标注数量: {results['statistics']['total_annotations']}")
    logger.info(f"类别数量: {results['statistics']['unique_classes']}")
    
    return results

def validate_annotation_file(file_path: str) -> Dict[str, any]:
    """
    验证单个标注文件
    
    Args:
        file_path: 标注文件路径
        
    Returns:
        验证结果
    """
    logger = logging.getLogger(__name__)
    file_path = Path(file_path)
    
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {
            'total_lines': 0,
            'valid_annotations': 0,
            'class_ids': set()
        }
    }
    
    if not file_path.exists():
        result['errors'].append(f"文件不存在: {file_path}")
        result['valid'] = False
        return result
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        result['statistics']['total_lines'] = len(lines)
        
        for line_no, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                result['errors'].append(f"行 {line_no}: 格式错误，需要至少5个值")
                continue
            
            try:
                class_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]
                
                result['statistics']['class_ids'].add(class_id)
                result['statistics']['valid_annotations'] += 1
                
                # 验证bbox
                x_center, y_center, width, height = bbox
                if not (0 <= x_center <= 1):
                    result['warnings'].append(f"行 {line_no}: x_center 超出范围 [0,1]: {x_center}")
                if not (0 <= y_center <= 1):
                    result['warnings'].append(f"行 {line_no}: y_center 超出范围 [0,1]: {y_center}")
                if not (0 < width <= 1):
                    result['warnings'].append(f"行 {line_no}: width 超出范围 (0,1]: {width}")
                if not (0 < height <= 1):
                    result['warnings'].append(f"行 {line_no}: height 超出范围 (0,1]: {height}")
                
                if class_id < 0:
                    result['errors'].append(f"行 {line_no}: 类别ID不能为负数: {class_id}")
                
            except ValueError as e:
                result['errors'].append(f"行 {line_no}: 数值转换错误: {e}")
    
    except Exception as e:
        result['errors'].append(f"读取文件失败: {e}")
    
    result['statistics']['class_ids'] = sorted(list(result['statistics']['class_ids']))
    result['valid'] = len(result['errors']) == 0
    
    return result

def check_class_distribution(dataset_path: str) -> Dict[str, any]:
    """
    分析数据集中的类别分布
    
    Args:
        dataset_path: 数据集路径
        
    Returns:
        类别分布统计
    """
    logger = logging.getLogger(__name__)
    dataset_path = Path(dataset_path)
    
    class_counts = Counter()
    file_class_counts = defaultdict(set)
    
    label_files = list(dataset_path.glob("**/*.txt"))
    
    for label_file in label_files:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        file_class_counts[class_id].add(str(label_file))
        
        except Exception as e:
            logger.warning(f"跳过文件 {label_file}: {e}")
    
    # 计算统计信息
    total_annotations = sum(class_counts.values())
    distribution = {}
    
    for class_id, count in class_counts.items():
        distribution[class_id] = {
            'count': count,
            'percentage': (count / total_annotations * 100) if total_annotations > 0 else 0,
            'files': len(file_class_counts[class_id])
        }
    
    result = {
        'total_annotations': total_annotations,
        'total_classes': len(class_counts),
        'distribution': distribution,
        'class_counts': dict(class_counts),
        'most_common': class_counts.most_common(),
        'least_common': class_counts.most_common()[::-1]
    }
    
    return result

def validate_image_label_pairs(dataset_path: str, 
                              images_dir: str = "images",
                              labels_dir: str = "labels") -> Dict[str, List[str]]:
    """
    验证图像和标签文件的配对关系
    
    Args:
        dataset_path: 数据集根路径
        images_dir: 图像目录名
        labels_dir: 标签目录名
        
    Returns:
        配对验证结果
    """
    dataset_path = Path(dataset_path)
    images_path = dataset_path / images_dir
    labels_path = dataset_path / labels_dir
    
    result = {
        'missing_labels': [],
        'missing_images': [],
        'valid_pairs': [],
        'statistics': {}
    }
    
    # 获取图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = {}
    
    if images_path.exists():
        for ext in image_extensions:
            for img_file in images_path.glob(f"*{ext}"):
                image_files[img_file.stem] = img_file
            for img_file in images_path.glob(f"*{ext.upper()}"):
                image_files[img_file.stem] = img_file
    
    # 获取标签文件
    label_files = {}
    if labels_path.exists():
        for label_file in labels_path.glob("*.txt"):
            label_files[label_file.stem] = label_file
    
    # 检查配对
    all_stems = set(image_files.keys()) | set(label_files.keys())
    
    for stem in all_stems:
        if stem in image_files and stem in label_files:
            result['valid_pairs'].append(stem)
        elif stem in image_files:
            result['missing_labels'].append(str(image_files[stem]))
        else:
            result['missing_images'].append(str(label_files[stem]))
    
    result['statistics'] = {
        'total_images': len(image_files),
        'total_labels': len(label_files),
        'valid_pairs': len(result['valid_pairs']),
        'missing_labels': len(result['missing_labels']),
        'missing_images': len(result['missing_images'])
    }
    
    return result
