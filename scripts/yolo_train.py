import os
import logging
from datetime import datetime
import shutil
from ultralytics import YOLO
import yaml
from pathlib import Path

def setup_logging(base_path, log_type='train', temp_log=True):
    log_dir = Path(base_path) / 'logging' / log_type
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f'temp-{timestamp}-{log_type}.log' if temp_log else f'{log_type}{timestamp}.log'
    log_file_path = log_dir / log_file_name
    logging.basicConfig(filename=log_file_path, level=logging.INFO, encoding='utf-8-sig',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return log_file_path

def load_model_and_config(base_path, model_name='yolov8n.pt', data_yaml='configs/data.yaml', epochs=50, batch=16, imgsz=640, lr0=0.01, device=''):
    model_path = Path(base_path) / 'models/pretrained' / model_name
    if not model_path.exists():
        logging.info(f"Model {model_name} not found. Downloading...")
        model = YOLO(model_name)
    else:
        model = YOLO(str(model_path))
    data_yaml_path = Path(base_path) / data_yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    return model, data_config

def train_model(model, data_config, base_path, epochs=50, batch=16, imgsz=640, lr0=0.01, device=''):
    results = model.train(data=str(Path(base_path) / 'configs/data.yaml'), epochs=epochs, batch=batch, imgsz=imgsz, lr0=lr0, device=device)
    return results

def save_model_weights(results, base_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_weight_path = results[0].best
    last_weight_path = results[0].last
    checkpoints_dir = Path(base_path) / 'models/checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_weight_destination = checkpoints_dir / f'trainN-{timestamp}-yolov8n-best.pt'
    last_weight_destination = checkpoints_dir / f'trainN-{timestamp}-yolov8n-last.pt'
    shutil.copy(best_weight_path, best_weight_destination)
    shutil.copy(last_weight_path, last_weight_destination)
    logging.info(f"Saved best weights to {best_weight_destination}")
    logging.info(f"Saved last weights to {last_weight_destination}")

def yolo_train():
    base_path = 'MedicalYOLO'
    log_file_path = setup_logging(base_path)
    model, data_config = load_model_and_config(base_path)
    logging.info("Training started")
    results = train_model(model, data_config, base_path)
    save_model_weights(results, base_path)
    logging.info(f"Training results: mAP@50={results[0].metrics.box.map50}, mAP@50:95={results[0].metrics.box.map}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_log_file_name = f'trainN-{timestamp}-yolov8n.log'
    new_log_file_path = Path(base_path) / 'logging' / 'train' / new_log_file_name
    os.rename(log_file_path, new_log_file_path)
    logging.info(f"Log renamed: {new_log_file_name}")

if __name__ == "__main__":
    yolo_train()