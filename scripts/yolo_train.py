#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@FileName : yolo_train.py
@Time     : 2025/6/26 13:00:05
@Author   : 雨滴同学
@Project  : BTD
@Function : 训练脚本的入口,集成utils模块
"""
import logging
from ultralytics import YOLO
import argparse
from pathlib import Path
import sys

current_path = Path(__file__).parent.parent.resolve()
utils_path = current_path / 'utils'
if str(current_path) not in sys.path:
    sys.path.insert(0, str(current_path))
if str(utils_path) not in sys.path:
    sys.path.insert(1, str(utils_path))

from utils.logging_utils import setup_logger
from utils.performance_utils import time_it
from utils.paths import LOGS_DIR, CHECKPOINTS_DIR, PRETRAINED_DIR
from utils.config_utils import load_config, merge_configs

def parse_args():
    """命名行解析参数"""
    parser = argparse.ArgumentParser(description="YOLO 模型训练")
    parser.add_argument("--data", type=str, default="data.yaml", help="yaml配置文件路径")
    parser.add_argument("--batch", type=int, default=16, help="训练批次大小")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="训练图片尺寸")
    parser.add_argument("--device", type=str, default="0", help="训练设备")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="预训练模型路径")
    parser.add_argument("--workers", type=int, default=8, help="训练数据加载线程数")
    # 自定义参数
    parser.add_argument("--use_yaml", type=bool, default=True, help="使用yaml配置文件")
    return parser.parse_args()

def run_training(model, yolo_args):
    result = model.train(**vars(yolo_args))
    return result

def main(args):
    logger = logging.getLogger("YOLO_Training")
    logger.info("YOLO 肿瘤检测训练脚本启动".center(80, "="))
    try:
        yaml_config = {}
        if args.use_yaml:
            yaml_config = load_config(config_type='train')

        # 合并参数
        yolo_args, project_args = merge_configs(
            mode='train',
            args=args,
            yaml_config=yaml_config,
            use_yaml=getattr(args, "use_yaml", True)
        )

        # 初始化模型,开始执行训练
        logger.info(f"初始化模型,加载权重: {project_args.weights}")
        model_path = PRETRAINED_DIR / project_args.weights
        if not model_path.exists():
            logger.info(f"模型文件不存在: {model_path},请将{project_args.weights}放入到{PRETRAINED_DIR}")
            raise ValueError(f"模型文件不存在: {model_path}")
        model = YOLO(model_path)

        # 动态引用 time_it 装饰器
        decorated_run_training = time_it(repeat_times=1, logger_instance=logger)(run_training)
        results = decorated_run_training(model, yolo_args)

        logger.info("YOLO 肿瘤检测训练脚本结束".center(80, "="))
    except Exception as e:
        logger.error(f"参数合并或训练异常: {e}")
        return

if __name__ == "__main__":
    args_ = parse_args()
    logger = setup_logger(
        base_path=LOGS_DIR,
        log_type="train",
        model_name=args_.weights.replace(".pt", ""),
        log_level=logging.INFO,
        temp_log=False,
        logger_name="YOLO_Training"
    )
    main(args_)