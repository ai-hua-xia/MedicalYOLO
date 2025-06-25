"""
数据集分割命令行接口
"""
import argparse
import shutil
import sys
import logging
from pathlib import Path
from typing import List
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.core import DatasetSplitter
from scripts.data_processing.utils import read_classes_file

def clean_previous_split(output_dir: Path):
    """清理旧的划分目录和 configs/data.yaml 文件"""
    for sub in ['train', 'val', 'test']:
        sub_dir = output_dir / sub
        if sub_dir.exists():
            shutil.rmtree(sub_dir)
            print(f"🧹 已删除目录: {sub_dir}")
    yaml_file = Path("configs/data.yaml")
    if yaml_file.exists():
        yaml_file.unlink()
        print(f"🧹 已删除配置文件: {yaml_file}")

def write_data_yaml(output_dir: Path, class_names: list):
    yaml_file = Path("configs/data.yaml")  # 固定输出到 configs/data.yaml
    yaml_content = f"""path: {output_dir.resolve()}
train: {output_dir.resolve() / 'train/images'}
val: {output_dir.resolve() / 'val/images'}
test: {output_dir.resolve() / 'test/images'}
nc: {len(class_names)}
names: {class_names}
"""
    yaml_file.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"📄 已生成配置文件: {yaml_file}")

def setup_logging(verbose: bool = False):
    """设置日志配置"""
    level = logging.DEBUG if verbose else logging.INFO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logging/dataset_split/dataset_split_{timestamp}.log'
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename)
        ]
    )

def validate_ratios(train: float, val: float, test: float) -> bool:
    """验证分割比例"""
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        print(f"❌ 分割比例总和必须为1.0，当前为: {total}")
        return False
    
    if train <= 0 or val < 0 or test < 0:
        print("❌ 分割比例必须为正数")
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MedicalYOLO 数据集分割工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 按比例分割数据集
  python split_cmd.py -i /path/to/dataset -o /path/to/output --train 0.7 --val 0.2 --test 0.1
  
  # 按类别平衡分割
  python split_cmd.py -i /path/to/dataset -o /path/to/output --by-class --train 0.8 --val 0.2
  
  # 生成YOLO配置文件
  python split_cmd.py -i /path/to/dataset -o /path/to/output --create-yaml --classes /path/to/classes.txt
        """
    )
    
    parser.add_argument(
        '-i', '--input',
    default='data/raw/images',
    help='输入图片路径 (默认: data/raw/images)'
    )
    
    parser.add_argument(
    '--labels',
    default='data/labels',
    help='标签文件目录 (默认: data/labels)'
    )
    
    parser.add_argument(
        '-o', '--output',
    default='data',
    help='输出根路径 (默认: data)'
    )
    
    parser.add_argument(
        '--train',
        type=float,
        default=0.8,
        help='训练集比例 (默认: 0.8)'
    )
    
    parser.add_argument(
        '--val',
        type=float,
        default=0.1,
        help='验证集比例 (默认: 0.1)'
    )
    
    parser.add_argument(
        '--test',
        type=float,
        default=0.1,
        help='测试集比例 (默认: 0.1)'
    )
    
    parser.add_argument(
        '--by-class',
        action='store_true',
        help='按类别平衡分割'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    
    parser.add_argument(
        '--image-extensions',
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
        help='支持的图像扩展名'
    )
    
    parser.add_argument(
        '--create-yaml',
        action='store_true',
        help='创建YOLO数据集配置文件'
    )
    
    parser.add_argument(
        '--classes',
        help='类别名称文件路径'
    )
    
    parser.add_argument(
        '--dataset-name',
        default='custom_dataset',
        help='数据集名称 (默认: custom_dataset)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # 验证分割比例
        if not validate_ratios(args.train, args.val, args.test):
            sys.exit(1)
        
        print(f"🚀 开始数据集分割...")
        print(f"   输入: {args.input}")
        print(f"   输出: {args.output}")
        print(f"   比例: 训练={args.train}, 验证={args.val}, 测试={args.test}")
        print(f"   方法: {'按类别平衡' if args.by_class else '随机分割'}")
        print(f"   种子: {args.seed}")
        
        # 初始化分割器
        splitter = DatasetSplitter(seed=args.seed)
        
        # 执行分割
        if args.by_class:
            results = splitter.split_by_class(
                args.input,
                args.output,
                args.train,
                args.val,
                args.test
            )
            
            print(f"\n📊 按类别分割结果:")
            for class_id, counts in results.items():
                print(f"   类别 {class_id}: 总计={counts['total']}, "
                      f"训练={counts['train']}, 验证={counts['val']}, 测试={counts['test']}")
        else:
            results = splitter.split_by_ratio(
                data_dir=args.input,
                labels_dir=args.labels,
                output_dir=args.output,
                train_ratio=args.train,
                val_ratio=args.val,
                test_ratio=args.test,
                image_extensions=args.image_extensions
            )
            
            print(f"\n📊 分割结果:")
            for split_name, counts in results.items():
                print(f"   {split_name}: 图像={counts['images']}, 标签={counts['labels']}")
        
        # 创建YOLO配置文件
        if args.create_yaml:
            class_names = []
            
            if args.classes:
                try:
                    class_names = read_classes_file(args.classes)
                    print(f"📋 从文件读取类别: {args.classes}")
                except Exception as e:
                    print(f"⚠️  无法读取类别文件: {e}")
            
            if not class_names:
                # 尝试从输出目录找到classes.txt
                classes_file = Path(args.output) / 'classes.txt'
                if classes_file.exists():
                    class_names = read_classes_file(str(classes_file))
                    print(f"📋 从输出目录读取类别: {classes_file}")
                else:
                    print("⚠️  未找到类别文件，使用默认类别名称")
                    class_names = ['class_0', 'class_1']  # 默认类别
            
            yaml_file = splitter.create_yolo_dataset_yaml(
                args.output,
                class_names,
                args.dataset_name
            )
            
            print(f"📄 YOLO配置文件: {yaml_file}")
        
        print(f"\n✅ 数据集分割完成!")
        print(f"   输出目录: {args.output}")
        
        # 显示输出目录结构
        output_path = Path(args.output)
        if output_path.exists():
            print(f"\n📁 输出目录结构:")
            for item in sorted(output_path.iterdir()):
                if item.is_dir():
                    image_count = len(list((item / 'images').glob('*'))) if (item / 'images').exists() else 0
                    label_count = len(list((item / 'labels').glob('*.txt'))) if (item / 'labels').exists() else 0
                    print(f"   {item.name}/")
                    print(f"     ├── images/ ({image_count} 文件)")
                    print(f"     └── labels/ ({label_count} 文件)")
                else:
                    print(f"   {item.name}")
        
        # 获取类别名
        class_names = []
        if args.classes:
            try:
                class_names = read_classes_file(args.classes)
            except Exception as e:
                print(f"⚠️  无法读取类别文件: {e}")
        if not class_names:
            # 自动查找
            candidates = [
                output_path / 'classes.txt',
                output_path / 'labels' / 'classes.txt'
            ]
            for classes_file in candidates:
                if classes_file.exists():
                    class_names = read_classes_file(str(classes_file))
                    print(f"📋 从 {classes_file} 读取类别")
                    break
            if not class_names:
                print("⚠️  未找到类别文件，使用默认类别名称")
                class_names = ['class_0', 'class_1']

        write_data_yaml(output_path, class_names)
        
        # 清理标签目录
        labels_dir = Path(args.labels)
        if labels_dir.exists():
            try:
                shutil.rmtree(labels_dir)
                print(f"🧹 已删除标签目录: {labels_dir}")
            except Exception as e:
                print(f"⚠️ 删除标签目录失败: {e}")
        
        logger.info("数据集分割完成")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 分割失败: {e}")
        logger.error(f"分割失败: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
