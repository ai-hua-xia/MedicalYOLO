#!/usr/bin/env python3
"""
运行COCO到YOLO转换器的脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入转换器
from scripts.data_processing.converters.coco_to_yolo import CocoToYoloConverter
from scripts.data_processing.utils.file_utils import move_files_by_extension, cleanup_temp_directory

def main():
    """主函数"""
    print("🚀 启动COCO到YOLO数据转换...")
    
    # 设置路径（使用绝对路径）
    project_root_str = str(project_root)
    input_path = os.path.join(project_root_str, "data/raw/annotations")
    output_path = os.path.join(project_root_str, "data/yolo_converted")
    
    print(f"项目根目录: {project_root_str}")
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_path):
        print(f"❌ 输入目录不存在: {input_path}")
        print("请确保您的COCO JSON文件放在 data/raw/annotations/ 目录下")
        
        # 尝试创建目录结构
        os.makedirs(input_path, exist_ok=True)
        print(f"✅ 已创建输入目录: {input_path}")
        print("请将您的COCO JSON文件放入此目录后重新运行")
        return
    
    try:
        # 初始化转换器
        converter = CocoToYoloConverter()
        
        # 执行转换
        result = converter.convert(input_path, output_path)
        
        print(f"\n🎉 转换成功完成!")
        print(f"📊 结果统计:")
        print(f"   - 转换标注数: {result['converted_annotations']}")
        print(f"   - 标签文件数: {result['converted_label_files']}")
        print(f"   - 类别数量: {len(result['class_names'])}")
        print(f"   - 类别列表: {result['class_names']}")
        
        # 移动文件到最终位置
        final_labels_dir = os.path.join(output_path, "labels")
        moved_count = move_files_by_extension(
            result['temp_dir'], 
            final_labels_dir, 
            ['txt']
        )
        print(f"📁 移动了 {moved_count} 个标签文件到: {final_labels_dir}")
        
        # 清理临时目录
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