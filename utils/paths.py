#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :paths.py
# @Time      :2025/6/23 14:30:00
# @Author    :雨霓同学
# @Project   :MedicalYOLO
# @Function  :项目路径配置

from pathlib import Path

# 项目根目录 - 需要向上一级才是真正的项目根目录
PROJECT_ROOT = Path(__file__).parent.parent  # 从 utils 目录向上一级

# 日志目录 - 统一使用 logging
LOGS_DIR = PROJECT_ROOT / "logging"

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"

# 模型目录
MODELS_DIR = PROJECT_ROOT / "models"

# 配置文件目录
CONFIGS_DIR = PROJECT_ROOT / "configs"

# 输出结果目录
OUTPUT_DIR = PROJECT_ROOT / "output"

# 临时文件目录
TEMP_DIR = PROJECT_ROOT / "temp"

# 运行文件目录
RUNS_DIR = PROJECT_ROOT / "runs"

# 确保关键目录存在
for directory in [LOGS_DIR, DATA_DIR, MODELS_DIR, CONFIGS_DIR, OUTPUT_DIR, TEMP_DIR, RUNS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)