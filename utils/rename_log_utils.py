import logging
from pathlib import Path

def rename_log_file(logger_obj, save_dir, model_name="model", encoding="utf-8"):
    """
    重命名日志文件，将临时日志文件重命名为更具描述性的名称，并切换日志写入新文件。
    :param logger_obj: logging.Logger 实例
    :param save_dir: 任务输出目录（如 runs/train/train1）
    :param model_name: 模型名或任务名
    :param encoding: 日志文件编码
    """
    for handler in list(logger_obj.handlers):
        if isinstance(handler, logging.FileHandler):
            old_log_file = Path(handler.baseFilename)
            # 解析时间戳
            # 假设原始文件名格式为 temp-YYYYMMDD-HHMMSS-xxx.log
            stem_parts = old_log_file.stem.split('-')
            if len(stem_parts) >= 3 and stem_parts[0] == "temp":
                timestamp = '-'.join(stem_parts[1:3])
            else:
                timestamp = "unknown"
            train_prefix = Path(save_dir).name
            new_log_file = old_log_file.parent / f"{train_prefix}_{timestamp}_{model_name}.log"

            handler.close()
            logger_obj.removeHandler(handler)

            if old_log_file.exists():
                try:
                    old_log_file.rename(new_log_file)
                    logger_obj.info(f"日志文件已成功重命名: {new_log_file}")
                except OSError as e:
                    logger_obj.error(f"日志文件重命名失败: {e}，日志仍写入原文件: {old_log_file}")
                    # 恢复旧处理器，保证日志不中断
                    new_handler = logging.FileHandler(old_log_file, encoding=encoding)
                    new_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                    logger_obj.addHandler(new_handler)
                    return old_log_file
            else:
                logger_obj.warning(f"待重命名的日志文件不存在: {old_log_file}")
                return None

            # 添加新的文件处理器
            new_handler = logging.FileHandler(new_log_file, encoding=encoding)
            new_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger_obj.addHandler(new_handler)
            return new_log_file
    logger_obj.warning("未找到 FileHandler，无法重命名日志文件。")
    return None