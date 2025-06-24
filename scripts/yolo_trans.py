import os
import json
import shutil
import random
import yaml
from pathlib import Path
import logging
from datetime import datetime

def setup_logging(base_path, log_type='data_conversion', temp_log=True):
    log_dir = Path(base_path) / 'logging' / log_type
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f'temp-{timestamp}-{log_type}.log' if temp_log else f'{log_type}{timestamp}.log'
    log_file_path = log_dir / log_file_name
    logging.basicConfig(filename=log_file_path, level=logging.INFO, encoding='utf-8-sig',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return log_file_path

def read_and_match_data(base_path):
    raw_images_dir = Path(base_path) / 'data/raw/images'
    raw_annotations_dir = Path(base_path) / 'data/raw/annotations'
    image_files = {f.stem: f for f in raw_images_dir.glob('*')}
    annotation_files = {f.stem: f for f in raw_annotations_dir.glob('*.json')}
    matched_files = []
    unmatched_images = []
    unmatched_annotations = []
    for stem in set(image_files.keys()) & set(annotation_files.keys()):
        matched_files.append((image_files[stem], annotation_files[stem]))
    for stem in set(image_files.keys()) - set(annotation_files.keys()):
        unmatched_images.append(image_files[stem])
    for stem in set(annotation_files.keys()) - set(image_files.keys()):
        unmatched_annotations.append(annotation_files[stem])
    logging.info(f"Matched {len(matched_files)} pairs of images and annotations")
    logging.info(f"Unmatched images: {len(unmatched_images)}")
    logging.info(f"Unmatched annotations: {len(unmatched_annotations)}")
    return matched_files

def convert_to_yolo_format(annotation, image_width, image_height):
    class_id = annotation['category_id']
    if 'bbox' in annotation:
        x, y, w, h = annotation['bbox']
        center_x = (x + w / 2) / image_width
        center_y = (y + h / 2) / image_height
        width = w / image_width
        height = h / image_height
        yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
    elif 'segmentation' in annotation:
        segmentation = annotation['segmentation'][0]
        normalized_segmentation = [coord / image_width if i % 2 == 0 else coord / image_height for i, coord in enumerate(segmentation)]
        yolo_line = f"{class_id} {' '.join([f'{coord:.6f}' for coord in normalized_segmentation])}"
    return yolo_line

def split_dataset(matched_files, base_path, split_ratio=(0.8, 0.1, 0.1)):
    random.seed(42)
    random.shuffle(matched_files)
    train_size = int(len(matched_files) * split_ratio[0])
    val_size = int(len(matched_files) * split_ratio[1])
    train_files = matched_files[:train_size]
    val_files = matched_files[train_size:train_size + val_size]
    test_files = matched_files[train_size + val_size:]
    for files, dataset_type in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
        dataset_dir = Path(base_path) / 'data' / dataset_type
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for image_file, annotation_file in files:
            shutil.copy(image_file, images_dir)
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            image_id_to_size = {image['id']: (image['width'], image['height']) for image in data['images']}
            annotations = {image['id']: [] for image in data['images']}
            for annotation in data['annotations']:
                annotations[annotation['image_id']].append(annotation)
            for image_id, image_annotations in annotations.items():
                image_width, image_height = image_id_to_size[image_id]
                label_lines = []
                for annotation in image_annotations:
                    yolo_line = convert_to_yolo_format(annotation, image_width, image_height)
                    label_lines.append(yolo_line)
                label_file_name = image_file.stem + '.txt'
                label_file_path = labels_dir / label_file_name
                with open(label_file_path, 'w') as f:
                    f.write('\n'.join(label_lines))
    logging.info(f"Dataset split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

def generate_data_yaml(base_path):
    data_yaml_path = Path(base_path) / 'configs' / 'data.yaml'
    categories = ['NO_tumor', 'glioma', 'meningioma', 'pituitary', 'space-occupying lesion']
    data_yaml = {
        'train': str(Path(base_path) / 'data/train/images'),
        'val': str(Path(base_path) / 'data/val/images'),
        'nc': len(categories),
        'names': categories
    }
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    logging.info(f"Generated data.yaml at {data_yaml_path}")

def yolo_trans():
    base_path = 'MedicalYOLO'
    log_file_path = setup_logging(base_path)
    matched_files = read_and_match_data(base_path)
    split_dataset(matched_files, base_path)
    generate_data_yaml(base_path)

if __name__ == "__main__":
    yolo_trans()