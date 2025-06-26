#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@FileName : config_utils.py
@Time     : 2025/06/26
@Author   : wyh
@Desc     : 加载配置文件，不存在时自动生成默认配置文件，防御性编程
"""

import logging
import yaml
import sys
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

# configs 目录在项目根目录下
CONFIGS_DIR = Path(__file__).parent.parent / "configs"
SUPPORTED_CONFIG_TYPES = {"train", "val", "infer"}

logger = logging.getLogger(__name__)

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

if __name__ == '__main__':
    # 示例：加载train配置
    config = load_config(config_type="train")