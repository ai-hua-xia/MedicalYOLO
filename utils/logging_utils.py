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

def log_parameters(args, exclude_params=None, logger=None):
    """
    记录命令行和YAML参数信息，返回结构化字典。

    Args:
        args: 命令行参数 (Namespace 对象)
        exclude_params: 不记录的参数键列表
        logger: 日志记录器实例

    Returns:
        dict: 参数字典
    """
    if logger is None:
        logger = logging.getLogger("YOLO_Training")
    if exclude_params is None:
        exclude_params = ['log_encoding', 'use_yaml', 'log_level', 'extra_args']
    logger.info("开始模型参数信息".center(40, "="))
    logger.info("Parameters")
    logger.info("-" * 40)
    params_dict = {}
    for key, value in vars(args).items():
        if key not in exclude_params and not key.endswith('_specified'):
            source = '命令行' if getattr(args, f"{key}_specified", False) else 'YAML'
            logger.info(f"{key:<20}: {value} （来源: [{source}]）")
            params_dict[key] = {"value": value, "source": source}
    return params_dict

def rename_log_file(logger_obj, save_dir, model_name, encoding="utf-8"):
    """
    主要实现日志的重命名,如train1, train2, train3....
    :param logger_obj: 日志记录器
    :param save_dir: 训练输出目录
    :param model_name: 模型名
    :param encoding: 文件编码
    :return: 新日志文件路径或None
    """
    for handler in list(logger_obj.handlers):
        if isinstance(handler, logging.FileHandler):
            old_log_file = Path(handler.baseFilename)
            # 解析时间戳
            timestamp = ""
            # 支持 temp-20250626-153000-xxx.log 或 xxx_20250626-153000.log
            if "-" in old_log_file.stem:
                timestamp_parts = old_log_file.stem.split("-")
                if len(timestamp_parts) >= 3 and timestamp_parts[0] == "temp":
                    timestamp = f"{timestamp_parts[1]}-{timestamp_parts[2]}"
                else:
                    timestamp = "-".join(timestamp_parts[1:3]) if len(timestamp_parts) > 2 else timestamp_parts[-1]
            elif "_" in old_log_file.stem:
                timestamp_parts = old_log_file.stem.split("_")
                if len(timestamp_parts) > 1:
                    timestamp = timestamp_parts[1]
            else:
                logger_obj.warning(
                    f"无法从日志文件名({old_log_file.name})中获取时间戳, "
                    f"请检查日志文件名是否正确"
                )
                continue
            train_prefix = Path(save_dir).name
            new_log_file = old_log_file.parent / f"{train_prefix}_{timestamp}_{model_name}.log"

            # 关闭旧的日志处理器
            handler.close()
            logger_obj.removeHandler(handler)

            if old_log_file.exists():
                try:
                    old_log_file.rename(new_log_file)
                    logger_obj.info(f"日志文件已经重命名成功: {new_log_file}")
                except OSError as e:
                    logger_obj.error(f"日志文件重命名失败: {e}")
                    # 恢复旧处理器，保证日志不中断
                    re_added_handler = logging.FileHandler(old_log_file, encoding=encoding)
                    re_added_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
                    logger_obj.addHandler(re_added_handler)
                    return old_log_file
            else:
                logger_obj.warning(f"尝试重命名的日志文件不存在: {old_log_file}")
                continue

            # 命名成功处理方案
            new_handler = logging.FileHandler(new_log_file, encoding=encoding)
            new_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger_obj.addHandler(new_handler)
            break
    else:
        logger_obj.warning("未找到 FileHandler，无法重命名日志文件。")
        return None
    return new_log_file

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