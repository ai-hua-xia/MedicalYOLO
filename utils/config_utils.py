#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@FileName : config_utils.py
@Time     : 2025/06/26
@Author   : wyh
@Desc     : 加载配置文件，不存在时自动生成默认配置文件，防御性编程，并实现参数合并
"""

import logging
import yaml
import sys
import copy
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.configs import (
    COMMENTED_TRAIN_CONFIG,
    COMMENTED_VAL_CONFIG,
    COMMENTED_INFER_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    DEFAULT_VAL_CONFIG,
    DEFAULT_INFER_CONFIG,
)
from utils.paths import CONFIGS_DIR, RUNS_DIR

logger = logging.getLogger(__name__)

SUPPORTED_CONFIG_TYPES = {"train", "val", "infer"}
YOLO_VALID_ARGS = {
    "train": set(DEFAULT_TRAIN_CONFIG.keys()),
    "val": set(DEFAULT_VAL_CONFIG.keys()),
    "infer": set(DEFAULT_INFER_CONFIG.keys()),
}

def generate_default_config(config_type: str):
    """
    生成默认的配置文件（内容来自configs.py）
    :param config_type: 配置文件类型
    """
    config_path = CONFIGS_DIR / f"{config_type}.yaml"
    if config_type == 'train':
        config = COMMENTED_TRAIN_CONFIG
    elif config_type == 'val':
        config = COMMENTED_VAL_CONFIG
    elif config_type == 'infer':
        config = COMMENTED_INFER_CONFIG
    else:
        logger.error(f"未知的配置文件类型: {config_type}")
        raise ValueError(f"配置文件类型错误: {config_type}, 目前仅支持train, val, infer三种模式")
    try:
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config)
        logger.info(f"生成默认 {config_type} 配置文件成功, 文件路径: {config_path}")
    except IOError as e:
        logger.error(f"写入默认 {config_type} 配置文件失败, 请检查文件权限和路径是否正确, 失败: {e}")
        raise
    except Exception as e:
        logger.error(f"生成配置文件 {config_path.name} 发生未知错误: {e}")
        raise

def load_config(config_type: str = 'train') -> dict:
    """
    加载配置文件, 若不存在则自动生成默认配置文件, 然后加载并返回
    :param config_type: 配置文件类型
    :return: 配置内容字典
    """
    config_path = CONFIGS_DIR / f"{config_type}.yaml"
    if not config_path.exists():
        logger.warning(f"配置文件({config_path})不存在, 尝试生成默认的配置文件")
        if config_type in SUPPORTED_CONFIG_TYPES:
            try:
                generate_default_config(config_type)
                logger.info(f"生成默认的配置文件成功: {config_path}")
            except Exception as e:
                logger.error(f"创建配置文件目录或生成失败: {e}")
                raise FileNotFoundError(f"创建配置文件目录失败: {e}")
        else:
            logger.error(f"配置文件类型错误: {config_type}")
            raise ValueError(f"配置文件类型错误: {config_type}, 目前仅支持train, val, infer三种模式")
    # 加载配置文件
    try:
        logger.info(f"正在加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"已加载配置文件: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"解析配置文件({config_path})失败: {e}")
        raise
    except Exception as e:
        logger.error(f"加载配置文件({config_path})失败: {e}")
        raise

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
    args=None,
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

    project_args = type('Args', (), {})()
    yolo_args = type('Args', (), {})()
    merged_params = copy.deepcopy(default_config)

    # 1. 合并 YAML 参数
    if use_yaml and yaml_config:
        for k, v in yaml_config.items():
            merged_params[k] = auto_type(v)

    # 2. 合并命令行参数（最高优先级）
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

    # 3. 路径标准化
    if "data" in merged_params and merged_params["data"]:
        data_path = Path(merged_params["data"])
        if not data_path.is_absolute():
            data_path = CONFIGS_DIR / data_path
        merged_params["data"] = str(data_path)
        if not data_path.exists():
            logger.warning(f"数据文件不存在: {data_path}")
    if "project" in merged_params and merged_params["project"]:
        project_path = Path(merged_params["project"])
        if not project_path.is_absolute():
            project_path = RUNS_DIR / project_path
        merged_params["project"] = str(project_path)
        try:
            project_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"无法创建项目目录: {project_path}, 错误: {e}")

    # 4. 分离 yolo_args 和 project_args
    for k, v in merged_params.items():
        setattr(project_args, k, v)
        if k in valid_args:
            setattr(yolo_args, k, v)
        if not hasattr(project_args, f"{k}_specified"):
            setattr(project_args, f"{k}_specified", False)

    # 5. 参数验证
    if mode == "train":
        if not isinstance(project_args.epochs, int) or project_args.epochs <= 0:
            raise ValueError("epochs 必须为正整数")
        if not isinstance(project_args.imgsz, int) or project_args.imgsz % 8 != 0:
            raise ValueError("imgsz 必须为8的倍数")
        if project_args.batch is not None and (not isinstance(project_args.batch, int) or project_args.batch <= 0):
            raise ValueError("batch 必须为正整数或 None")
        if not Path(project_args.data).exists():
            raise ValueError(f"data 文件不存在: {project_args.data}")

    return yolo_args, project_args

if __name__ == '__main__':
    # 示例：加载train配置
    config = load_config(config_type="train")