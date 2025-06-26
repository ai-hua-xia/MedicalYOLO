import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.config_utils import load_config, merge_configs

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 配置加载与合并工具")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'infer'], help='配置模式')
    parser.add_argument('--config', type=str, default=None, help='自定义配置文件路径（可选）')
    parser.add_argument('--extra_args', nargs='*', default=None, help='额外参数，格式: key value ...')
    # 你可以根据需要添加更多参数
    return parser.parse_args()

def main():
    args = parse_args()
    # 加载配置文件
    if args.config:
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
    else:
        yaml_config = load_config(args.mode)
    # 合并参数
    yolo_args, project_args = merge_configs(args.mode, args=args, yaml_config=yaml_config)
    print("YOLO 参数：")
    for k, v in vars(yolo_args).items():
        print(f"  {k}: {v}")
    print("\n项目参数：")
    for k, v in vars(project_args).items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()