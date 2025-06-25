#!/usr/bin/env python3
"""
统一数据转换运行脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.data_processing.core.data_converter import DataConverter
from scripts.data_processing.utils.file_utils import move_files_by_extension, cleanup_temp_directory

def main():
    print("🚀 启动数据格式转换...")

    # 设置路径
    input_path = os.path.join(str(project_root), "data/raw/annotations")
    output_path = os.path.join(str(project_root), "data/yolo_converted")

    print(f"项目根目录: {project_root}")
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")

    # 检查输入目录是否存在
    if not os.path.exists(input_path):
        print(f"❌ 输入目录不存在: {input_path}")
        os.makedirs(input_path, exist_ok=True)
        print(f"✅ 已创建输入目录: {input_path}")
        print("请将您的COCO JSON文件放入此目录后重新运行")
        return

    try:
        # 统一入口
        data_converter = DataConverter()
        # 这里指定转换类型
        conversion_type = "coco_to_yolo"
        result = data_converter.convert(
            conversion_type=conversion_type,
            input_path=input_path,
            output_path=output_path
        )

        print(f"\n🎉 转换成功完成!")
        print(f"📊 结果统计:")
        print(f"   - 转换标注数: {result['converted_annotations']}")
        print(f"   - 标签文件数: {result['converted_label_files']}")
        print(f"   - 类别数量: {len(result['class_names'])}")
        print(f"   - 类别列表: {result['class_names']}")

        final_labels_dir = os.path.join(output_path, "labels")
        moved_count = move_files_by_extension(
            result['temp_dir'],
            final_labels_dir,
            ['txt']
        )
        print(f"📁 移动了 {moved_count} 个标签文件到: {final_labels_dir}")

        if cleanup_temp_directory(result['temp_dir']):
            print(f"🧹 临时目录清理完成")

        print(f"\n✅ 所有操作完成!")
        print(f"   输出目录: {output_path}")
        print(f"   标签目录: {final_labels_dir}")
        print(f"   类别文件: {os.path.join(output_path, 'classes.txt')}")

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()