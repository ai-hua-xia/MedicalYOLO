"""
数据转换命令行接口
"""
import argparse
from datetime import datetime
import sys
import logging
from pathlib import Path
from typing import Optional, Dict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.core import DataConverter
from scripts.data_processing.utils import read_json_file

def setup_logging(verbose: bool = False):
    """设置日志配置"""
    level = logging.DEBUG if verbose else logging.INFO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logging/data_conversion/data_conversion_{timestamp}.log'
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename)
        ]
    )

def parse_class_mapping(mapping_file: Optional[str]) -> Optional[Dict[str, int]]:
    """解析类别映射文件"""
    if not mapping_file:
        return None
    
    try:
        mapping_data = read_json_file(mapping_file)
        return mapping_data
    except Exception as e:
        print(f"❌ 无法读取类别映射文件 {mapping_file}: {e}")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MedicalYOLO 数据格式转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # COCO转YOLO
  python convert_cmd.py coco_to_yolo -i /path/to/coco/annotations -o /path/to/yolo/output
  
  # Pascal VOC转YOLO
  python convert_cmd.py pascal_to_yolo -i /path/to/pascal/annotations -o /path/to/yolo/output
  
  # LabelMe转YOLO
  python convert_cmd.py labelme_to_yolo -i /path/to/labelme/json -o /path/to/yolo/output --format detection
        """
    )
    
    parser.add_argument(
        'conversion_type',
        choices=['coco_to_yolo', 'pascal_to_yolo', 'labelme_to_yolo'],
        help='转换类型'
    )
    
    parser.add_argument(
        '-i', '--input',
        default='data/raw/annotations',
        help='输入数据路径(默认: data/raw/annotations)'
    )
    
    parser.add_argument(
    '--images',
    default='data/raw/images',
    help='输入图片路径 (默认: data/raw/images)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='data/labels',
        help='输出标签路径 (默认: data/labels)'
    )
    
    parser.add_argument(
        '-m', '--mapping',
        help='类别映射JSON文件路径'
    )
    
    parser.add_argument(
        '--format',
        choices=['detection', 'segmentation'],
        default='detection',
        help='输出格式 (仅对LabelMe有效)'
    )
    
    parser.add_argument(
        '--include-difficult',
        action='store_true',
        help='包含difficult标记的对象 (仅对Pascal VOC有效)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='转换前验证输入格式'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        print(f"🚀 开始 {args.conversion_type} 转换...")
        print(f"   输入: {args.input}")
        print(f"   输出: {args.output}")
        
        # 初始化转换器
        converter = DataConverter()
        
        # 验证输入（如果需要）
        if args.validate:
            print("🔍 验证输入格式...")
            if not converter.validate_input(args.conversion_type, args.input):
                print("❌ 输入验证失败")
                sys.exit(1)
            print("✅ 输入验证通过")
        
        # 解析类别映射
        class_mapping = parse_class_mapping(args.mapping)
        if class_mapping:
            print(f"📋 使用类别映射: {class_mapping}")
        
        # 准备转换参数
        convert_kwargs = {}
        
        if args.conversion_type == 'labelme_to_yolo':
            convert_kwargs['output_format'] = args.format
        
        if args.conversion_type == 'pascal_to_yolo':
            convert_kwargs['include_difficult'] = args.include_difficult
        
        # 执行转换
        result = converter.convert(
            args.conversion_type,
            args.input,
            args.output,
            class_mapping,
            **convert_kwargs
        )
        
        # 显示结果
        print(f"\n🎉 转换完成!")
        print(f"📊 转换结果:")
        
        if 'converted_label_files' in result:
            print(f"   - 标签文件数: {result['converted_label_files']}")
        
        if 'converted_annotations' in result:
            print(f"   - 标注数量: {result['converted_annotations']}")
        
        if 'class_names' in result:
            print(f"   - 类别数量: {len(result['class_names'])}")
            print(f"   - 类别列表: {result['class_names']}")
        
        if 'skipped_annotations' in result:
            print(f"   - 跳过标注: {result['skipped_annotations']}")
        
        print(f"\n✅ 输出目录: {args.output}")
        
        if 'temp_dir' in result:
            from pathlib import Path
            temp_dir = Path(result['temp_dir'])
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            moved = 0
            for txt_file in temp_dir.glob('*.txt'):
                target = output_dir / txt_file.name
                txt_file.replace(target)
                moved += 1
            print(f"📁 已移动 {moved} 个标签文件到 {output_dir}")
            # 不要清理 temp_dir，等分割后再清理
        
        logger.info(f"转换完成: {args.conversion_type}")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        logger.error(f"转换失败: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
