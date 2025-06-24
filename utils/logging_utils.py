#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :logging_utils.py
# @Time      :2025/6/23 14:28:17
# @Author    :雨霓同学
# @Project   :MedicalYOLO
# @Function  :日志相关的工具类函数
import logging
from datetime import datetime
from pathlib import Path


def setup_logger(base_path: Path, log_type: str = "general",
                 model_name: str = None,
                 encoding: str = "utf-8",
                 log_level: int = logging.INFO,
                 temp_log: bool = False,
                 logger_name: str = "YOLO Default"
                 ):
    """
    配置日志记录器，将日志保存到指定路径的子目录当中，并同时输出到控制台，日志文件名为类型 + 时间戳
    :param base_path: 日志文件的根路径
    :param log_type: 日志的类型
    :param encoding: 文件编码
    :param log_level: 日志等级
    :param temp_log: 是否启动临时文件名
    :param logger_name: 日志记录器的名称
    :return: logging.logger: 返回一个日志记录器实例
    """
    # 1. 构建日志文件完整的存放路径
    log_dir = base_path / log_type
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2. 生成一个带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # 根据temp_log参数，生成不同的日志文件名前缀
    prefix = "temp" if temp_log else log_type.replace(" ", "-")
    log_filename_parts = [prefix, timestamp]
    if model_name:
        log_filename_parts.append(model_name.replace(" ", "-"))
    log_filename = "_".join(log_filename_parts) + ".log"
    log_file = log_dir / log_filename

    # 3. 获取或创建指定的名称logger实例
    logger = logging.getLogger(logger_name)
    # 设定日志记录器记录最低级别
    logger.setLevel(log_level)
    # 阻止日志事件传播到父级logger
    logger.propagate = False

    # 4. 需要避免重复添加日志处理器，因此先检查日志处理器列表中是否已经存在了指定的日志处理器
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # 5.创建文件处理器，将日志写入到文件当中
    file_handler = logging.FileHandler(log_file, encoding=encoding)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s : %(message)s")
    )
    # 将文件处理器添加到logger实例中
    logger.addHandler(file_handler)

    # 6.创建控制台处理器，将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s : %(message)s")
    )
    # 将控制台处理器添加到logger实例中
    logger.addHandler(console_handler)

    # 输出一些初始化信息到日志，确认配置成功
    logger.info(f"日志记录器已启动，日志文件保存在: {log_file}")
    logger.info(f"日志记录器的根目录: {base_path}")
    logger.info(f"日志记录器的名称: {logger_name}")
    logger.info(f"日志记录器的类型: {log_type}")
    logger.info(f"日志记录器的级别: {logging.getLevelName(log_level)}")
    logger.info("日志记录器初始化成功".center(60, "="))
    return logger


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # 添加项目根目录到Python路径
    sys.path.append(str(Path(__file__).parent.parent))
    
    from utils.paths import LOGS_DIR
    logger = setup_logger(base_path=LOGS_DIR,
                          log_type="test_log", model_name=None,
                          )
    logger.info("测试日志记录器")