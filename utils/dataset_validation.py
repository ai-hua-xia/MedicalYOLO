import yaml
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any
import shutil

def verify_dataset_config(yaml_path: Path, current_logger: logging.Logger, mode: str, task_type: str) -> Tuple[bool, List[Dict]]:
    """
    验证 data.yaml 配置和数据集内容
    """
    invalid_data = []
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f)
    except Exception as e:
        current_logger.error(f"无法读取yaml文件: {e}")
        return False, [{"image_path": None, "label_path": None, "error_message": f"无法读取yaml文件: {e}"}]

    # 检查类别数和类别名
    nc = data_cfg.get('nc')
    names = data_cfg.get('names')
    if not isinstance(names, list) or len(names) != nc:
        msg = f"类别数(nc)与类别名(names)数量不一致: nc={nc}, names={names}"
        current_logger.error(msg)
        invalid_data.append({"image_path": None, "label_path": None, "error_message": msg})

    # 检查每个分割
    for split in ['train', 'val', 'test']:
        split_path = data_cfg.get(split)
        if not split_path:
            current_logger.warning(f"{split} 路径未在yaml中定义，跳过")
            continue
        img_dir = Path(split_path)
        if not img_dir.exists() or not img_dir.is_dir():
            msg = f"{split} 图像目录不存在: {img_dir}"
            current_logger.error(msg)
            invalid_data.append({"image_path": str(img_dir), "label_path": None, "error_message": msg})
            continue
        img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        if not img_files:
            msg = f"{split} 图像目录为空: {img_dir}"
            current_logger.error(msg)
            invalid_data.append({"image_path": str(img_dir), "label_path": None, "error_message": msg})
            continue

        # 抽样或全量
        check_files = img_files
        if mode.upper() == "SAMPLE" and len(img_files) > 20:
            import random
            check_files = random.sample(img_files, 20)

        for img_path in check_files:
            label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            if not label_path.exists():
                msg = f"标签文件不存在: {label_path}"
                current_logger.error(msg)
                invalid_data.append({"image_path": str(img_path), "label_path": str(label_path), "error_message": msg})
                continue
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
            except Exception as e:
                msg = f"标签文件读取失败: {label_path}, 错误: {e}"
                current_logger.error(msg)
                invalid_data.append({"image_path": str(img_path), "label_path": str(label_path), "error_message": msg})
                continue

            for idx, line in enumerate(lines):
                parts = line.split()
                if task_type == "detection":
                    if len(parts) != 5:
                        msg = f"检测任务标签格式错误(应为5项): {line}"
                        current_logger.error(msg)
                        invalid_data.append({"image_path": str(img_path), "label_path": str(label_path), "error_message": msg})
                        continue
                elif task_type == "segmentation":
                    if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                        msg = f"分割任务标签格式错误: {line}"
                        current_logger.error(msg)
                        invalid_data.append({"image_path": str(img_path), "label_path": str(label_path), "error_message": msg})
                        continue
                # 检查数值
                try:
                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= nc:
                        msg = f"类别ID超出范围: {class_id}"
                        current_logger.error(msg)
                        invalid_data.append({"image_path": str(img_path), "label_path": str(label_path), "error_message": msg})
                        continue
                    coords = list(map(float, parts[1:]))
                    if task_type == "detection":
                        if not all(0.0 <= v <= 1.0 for v in coords):
                            msg = f"检测坐标超出[0,1]范围: {coords}"
                            current_logger.error(msg)
                            invalid_data.append({"image_path": str(img_path), "label_path": str(label_path), "error_message": msg})
                    elif task_type == "segmentation":
                        if not all(0.0 <= v <= 1.0 for v in coords):
                            msg = f"分割坐标超出[0,1]范围: {coords}"
                            current_logger.error(msg)
                            invalid_data.append({"image_path": str(img_path), "label_path": str(label_path), "error_message": msg})
                except Exception as e:
                    msg = f"标签内容解析失败: {line}, 错误: {e}"
                    current_logger.error(msg)
                    invalid_data.append({"image_path": str(img_path), "label_path": str(label_path), "error_message": msg})

    passed = len(invalid_data) == 0
    if passed:
        current_logger.info("数据集验证通过！")
    else:
        current_logger.warning(f"数据集验证未通过，共发现 {len(invalid_data)} 个问题样本。")
    return passed, invalid_data

def verify_split_uniqueness(yaml_path: Path, current_logger: logging.Logger) -> bool:
    """
    检查train/val/test分割之间是否有重复图片
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f)
    except Exception as e:
        current_logger.error(f"无法读取yaml文件: {e}")
        return False

    split_imgs = {}
    for split in ['train', 'val', 'test']:
        split_path = data_cfg.get(split)
        if not split_path:
            continue
        img_dir = Path(split_path)
        if not img_dir.exists() or not img_dir.is_dir():
            continue
        img_files = set(p.name for p in img_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png'])
        split_imgs[split] = img_files

    # 检查重复
    all_names = []
    for split in split_imgs:
        all_names.extend([(split, name) for name in split_imgs[split]])
    name_to_splits = {}
    for split, name in all_names:
        name_to_splits.setdefault(name, []).append(split)
    duplicates = {name: splits for name, splits in name_to_splits.items() if len(splits) > 1}
    if duplicates:
        for name, splits in duplicates.items():
            current_logger.error(f"图片 {name} 同时出现在 {splits}")
        return False
    current_logger.info("train/val/test 分割唯一性验证通过！")
    return True

def delete_invalid_files(invalid_data_list: list, current_logger: logging.Logger):
    """
    删除不合法的图片和标签文件
    """
    for item in invalid_data_list:
        img_path = item.get('image_path')
        label_path = item.get('label_path')
        for path in [img_path, label_path]:
            if path and Path(path).exists():
                try:
                    Path(path).unlink()
                    current_logger.warning(f"已删除文件: {path}")
                except Exception as e:
                    current_logger.error(f"删除文件失败: {path}, 错误: {e}")