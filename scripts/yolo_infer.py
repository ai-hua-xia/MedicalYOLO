import os
import logging
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path

def setup_logging(base_path, log_type='infer', temp_log=True):
    log_dir = Path(base_path) / 'logging' / log_type
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f'temp-{timestamp}-{log_type}.log' if temp_log else f'{log_type}{timestamp}.log'
    log_file_path = log_dir / log_file_name
    logging.basicConfig(filename=log_file_path, level=logging.INFO, encoding='utf-8-sig',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return log_file_path

def load_model(base_path, weights='models/checkpoints/trainN-20250614_200001-yolov8n-best.pt'):
    weights_path = Path(base_path) / weights
    model = YOLO(str(weights_path))
    return model

def infer_model(model, source, base_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    infer_dir = Path(base_path) / 'runs/infer' / f'inferN-{timestamp}'
    infer_dir.mkdir(parents=True, exist_ok=True)
    results = model(source, save=True, save_txt=True, project=str(infer_dir))
    return results

def yolo_infer():
    base_path = 'MedicalYOLO'
    log_file_path = setup_logging(base_path)
    model = load_model(base_path)
    source = 'data/raw/images'
    logging.info(f"Inference started on {source}")
    results = infer_model(model, source, base_path)
    logging.info(f"Inference completed. Results saved to {results[0].save_dir}")

if __name__ == "__main__":
    yolo_infer()