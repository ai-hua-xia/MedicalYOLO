import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.rename_log_utils import rename_log_file

# 日志初始化（模拟 setup_logging，生成临时日志文件）
LOG_DIR = Path(__file__).parent.parent / "logging" / "load_yaml"
LOG_DIR.mkdir(parents=True, exist_ok=True)
temp_log_file = LOG_DIR / "temp-20250626-153000-demo.log"

logger = logging.getLogger("yolo_logger")
logger.setLevel(logging.INFO)
# 只添加一次handler，避免重复
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    file_handler = logging.FileHandler(temp_log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

logger.info("这是临时日志文件，准备重命名...")

# 假设任务输出目录和模型名
save_dir = Path("runs/train/train1")
model_name = "yolov8"

# 调用重命名工具
new_log_path = rename_log_file(logger, save_dir, model_name)
if new_log_path:
    logger.info(f"日志重命名完成，新日志路径: {new_log_path}")
else:
    logger.error("日志重命名失败。")