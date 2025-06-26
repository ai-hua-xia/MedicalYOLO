import argparse
import copy
import logging
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))  # 添加 utils 目录到路径

from utils.config_utils import load_config
from utils.configs import DEFAULT_TRAIN_CONFIG, DEFAULT_VAL_CONFIG, DEFAULT_INFER_CONFIG
from utils.paths import CONFIGS_DIR, RUNS_DIR

logger = logging.getLogger(__name__)

# 假设你有如下模式参数集合
YOLO_VALID_ARGS = {
    "train": set(DEFAULT_TRAIN_CONFIG.keys()),
    "val": set(DEFAULT_VAL_CONFIG.keys()),
    "infer": set(DEFAULT_INFER_CONFIG.keys()),
}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("yes", "true", "t", "1")
    return bool(v)

def auto_type(val):
    # 尝试将字符串转换为 int、float、bool、None 或列表
    if isinstance(val, str):
        if val.lower() == "none":
            return None
        if val.lower() in ("true", "false"):
            return str2bool(val)
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                if "," in val:
                    return [auto_type(x.strip()) for x in val.split(",")]
    return val

def merge_configs(
    mode: str,
    args: argparse.Namespace = None,
    yaml_config: dict = None,
    use_yaml: bool = True
):
    """
    参数合并：默认值 < YAML < 命令行
    :param mode: 'train'/'val'/'infer'
    :param args: argparse.Namespace，命令行参数
    :param yaml_config: dict，YAML 文件参数
    :param use_yaml: 是否使用yaml参数
    :return: (yolo_args, project_args)
    """
    # 1. 确定模式和默认参数
    if mode not in YOLO_VALID_ARGS:
        raise ValueError(f"不支持的模式: {mode}")
    valid_args = YOLO_VALID_ARGS[mode]
    if mode == "train":
        default_config = copy.deepcopy(DEFAULT_TRAIN_CONFIG)
    elif mode == "val":
        default_config = copy.deepcopy(DEFAULT_VAL_CONFIG)
    elif mode == "infer":
        default_config = copy.deepcopy(DEFAULT_INFER_CONFIG)
    else:
        raise NotImplementedError(f"暂未实现 {mode} 的默认参数")

    # 2. 初始化参数存储
    project_args = argparse.Namespace()
    yolo_args = argparse.Namespace()
    merged_params = copy.deepcopy(default_config)

    # 3. 合并 YAML 参数
    if use_yaml and yaml_config:
        for k, v in yaml_config.items():
            merged_params[k] = auto_type(v)

    # 4. 合并命令行参数（最高优先级）
    if args is not None:
        for k, v in vars(args).items():
            if v is not None and k != "extra_args":
                merged_params[k] = auto_type(v)
                setattr(project_args, f"{k}_specified", True)
        # 处理 extra_args
        if hasattr(args, "extra_args") and args.extra_args:
            extra = args.extra_args
            if len(extra) % 2 != 0:
                raise ValueError("extra_args 必须为键值对")
            for i in range(0, len(extra), 2):
                key, value = extra[i], extra[i + 1]
                merged_params[key] = auto_type(value)
                setattr(project_args, f"{key}_specified", True)

    # 5. 路径标准化
    # data 路径
    if "data" in merged_params and merged_params["data"]:
        data_path = Path(merged_params["data"])
        if not data_path.is_absolute():
            data_path = CONFIGS_DIR / data_path
        merged_params["data"] = str(data_path)
        if not data_path.exists():
            logger.warning(f"数据文件不存在: {data_path}")
    # project 路径
    if "project" in merged_params and merged_params["project"]:
        project_path = Path(merged_params["project"])
        if not project_path.is_absolute():
            project_path = RUNS_DIR / project_path
        merged_params["project"] = str(project_path)
        try:
            project_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"无法创建项目目录: {project_path}, 错误: {e}")

    # 6. 分离 yolo_args 和 project_args
    for k, v in merged_params.items():
        setattr(project_args, k, v)
        if k in valid_args:
            setattr(yolo_args, k, v)
        # 来源标记
        if not hasattr(project_args, f"{k}_specified"):
            setattr(project_args, f"{k}_specified", False)

    # 7. 参数验证
    if mode == "train":
        if not isinstance(project_args.epochs, int) or project_args.epochs <= 0:
            raise ValueError("epochs 必须为正整数")
        if not isinstance(project_args.imgsz, int) or project_args.imgsz % 8 != 0:
            raise ValueError("imgsz 必须为8的倍数")
        if project_args.batch is not None and (not isinstance(project_args.batch, int) or project_args.batch <= 0):
            raise ValueError("batch 必须为正整数或 None")
        if not Path(project_args.data).exists():
            raise ValueError(f"data 文件不存在: {project_args.data}")

    # 其他模式可补充

    return yolo_args, project_args