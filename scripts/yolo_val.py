import os
import logging
from datetime import datetime
from ultralytics import YOLO
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

def setup_logging(base_path, log_type='val', temp_log=True):
    log_dir = Path(base_path) / 'logging' / log_type
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f'temp-{timestamp}-{log_type}.log' if temp_log else f'{log_type}{timestamp}.log'
    log_file_path = log_dir / log_file_name
    logging.basicConfig(filename=log_file_path, level=logging.INFO, encoding='utf-8-sig',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return log_file_path

def load_model_and_config(base_path, weights='models/checkpoints/trainN-20250614_200001-yolov8n-best.pt', data_yaml='configs/data.yaml', imgsz=640, device=''):
    weights_path = Path(base_path) / weights
    model = YOLO(str(weights_path))
    data_yaml_path = Path(base_path) / data_yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    return model, data_config

def validate_model(model, data_config, base_path, imgsz=640, device=''):
    results = model.val(data=str(Path(base_path) / 'configs/data.yaml'), imgsz=imgsz, device=device)
    return results

def visualize_time_stats(results, base_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validation_dir = Path(base_path) / 'runs/val' / f'validationN-{timestamp}'
    validation_dir.mkdir(parents=True, exist_ok=True)
    time_stats = results[0].speed
    plt.plot(time_stats['preprocess'], label='Preprocess')
    plt.plot(time_stats['inference'], label='Inference')
    plt.plot(time_stats['postprocess'], label='Postprocess')
    plt.xlabel('Batch')
    plt.ylabel('Time (ms)')
    plt.legend()
    time_stats_path = validation_dir / 'time_stats.png'
    plt.savefig(time_stats_path)
    logging.info(f"Saved time stats visualization to {time_stats_path}")

def yolo_val():
    base_path = 'MedicalYOLO'
    log_file_path = setup_logging(base_path)
    model, data_config = load_model_and_config(base_path)
    logging.info("Validation started")
    results = validate_model(model, data_config, base_path)
    logging.info(f"Validation results: mAP@50={results[0].metrics.box.map50}, mAP@50:95={results[0].metrics.box.map}, Precision={results[0].metrics.box.p}, Recall={results[0].metrics.box.r}")
    print(f"mAP@50={results[0].metrics.box.map50}, mAP@50:95={results[0].metrics.box.map}, Precision={results[0].metrics.box.p}, Recall={results[0].metrics.box.r}")
    visualize_time_stats(results, base_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_log_file_name = f'valN-{timestamp}-yolov8n.log'
    new_log_file_path = Path(base_path) / 'logging' / 'val' / new_log_file_name
    os.rename(log_file_path, new_log_file_path)
    logging.info(f"Log renamed: {new_log_file_name}")

if __name__ == "__main__":
    yolo_val()