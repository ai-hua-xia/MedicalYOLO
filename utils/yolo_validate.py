import os
import argparse
import logging
from pathlib import Path
from dataset_validation import (
    verify_dataset_config,
    verify_split_uniqueness,
    delete_invalid_files
)

def setup_logger(logfile: str = "logging/dataset_validation/yolo_validate.log"):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)  # 确保日志目录存在
    logger = logging.getLogger("YOLOValidate")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh = logging.FileHandler(logfile, encoding='utf-8')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def main():
    parser = argparse.ArgumentParser(description="YOLO数据集验证工具")
    parser.add_argument('--yaml', type=str, default='configs/data.yaml', help='data.yaml 路径')
    parser.add_argument('--mode', type=str, default='FULL', choices=['FULL', 'SAMPLE'], help='验证模式')
    parser.add_argument('--task', type=str, default='detection', choices=['detection', 'segmentation'], help='任务类型')
    parser.add_argument('--delete-invalid', action='store_true', help='验证失败时自动删除不合法文件')
    args = parser.parse_args()

    logger = setup_logger()

    logger.info(f"开始验证: {args.yaml}，模式: {args.mode}，任务: {args.task}")

    passed, invalid_data = verify_dataset_config(
        yaml_path=Path(args.yaml),
        current_logger=logger,
        mode=args.mode,
        task_type=args.task
    )

    if not passed:
        logger.warning(f"数据集基础验证未通过，共 {len(invalid_data)} 个问题样本。")
        if args.delete_invalid and invalid_data:
            logger.warning("即将删除所有不合法的图片和标签文件...")
            delete_invalid_files(invalid_data, logger)
            logger.info("不合法文件已删除。")
        else:
            logger.info("未启用自动删除，请手动处理不合法文件。")

    unique = verify_split_uniqueness(
        yaml_path=Path(args.yaml),
        current_logger=logger
    )
    if not unique:
        logger.warning("train/val/test 分割存在重复图片，请检查！")

    if passed and unique:
        logger.info("🎉 数据集验证全部通过！")
    else:
        logger.warning("❌ 数据集验证未全部通过，请根据日志修复问题。")

if __name__ == "__main__":
    main()