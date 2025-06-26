import sys
import shutil
from pathlib import Path
import subprocess

def main():
    project_root = Path(__file__).resolve().parents[3]
    convert_cmd = [
        sys.executable,
        str(project_root / "scripts/data_processing/cli/convert_cmd.py"),
        "--conversion_type", "coco_to_yolo",
        "-i", "data/raw/annotations",
        "-o", "data/labels"
    ]
    split_cmd = [
        sys.executable,
        str(project_root / "scripts/data_processing/cli/split_cmd.py"),
        "-i", "data/raw/images",
        "-o", "data"
    ]

    print("🚀 开始数据格式转换 ...")
    result1 = subprocess.run(convert_cmd)
    if result1.returncode != 0:
        print("❌ 数据格式转换失败，流程终止。")
        sys.exit(1)

    print("🚀 开始数据集划分 ...")
    result2 = subprocess.run(split_cmd)
    if result2.returncode != 0:
        print("❌ 数据集划分失败，流程终止。")
        sys.exit(1)

    print("✅ 全部流程完成！")

if __name__ == "__main__":
    main()