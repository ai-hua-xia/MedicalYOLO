import logging
from pathlib import Path

def rename_log_file(logger_obj, save_dir, model_name, encoding="utf-8"):
    """
    主要实现日志的重命名,如train1, train2, train3....
    :param logger_obj:
    :param save_dir:
    :param model_name:
    :param encoding:
    :return:
    """
    # 遍历当前的日志记录器
    for handler in list(logger_obj.handlers):
        if isinstance(handler, logging.FileHandler):
            old_log_file = Path(handler.baseFilename)
            # 解析时间戳
            timestamp = ""
            timestamp_parts = old_log_file.stem.split("-")
            if len(timestamp_parts) >= 3 and timestamp_parts[0] == "temp":
                timestamp = f"{timestamp_parts[1]}-{timestamp_parts[2]}"
            elif len(old_log_file.stem.split("_")) > 1:
                timestamp = old_log_file.stem.split("_")[1]
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