import os
import shutil
from pathlib import Path
import logging
import time
from datetime import datetime
from performance_utils import time_it


def setup_logging(base_path, log_type='project_init', temp_log=True):
    log_dir = Path(base_path) / 'logging' / log_type
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f'temp-{timestamp}-{log_type}.log' if temp_log else f'{log_type}_{timestamp}.log'
    log_file_path = log_dir / log_file_name
    
    # 清除之前的日志配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        filename=log_file_path, 
        level=logging.INFO, 
        encoding='utf-8',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    
    # 同时输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)
    
    return log_file_path


@time_it
def create_directory_structure(base_path):
    """创建项目目录结构"""
    directories = [
        'configs',
        'data/raw/images',
        'data/raw/annotations',
        'data/train/images',
        'data/train/labels',
        'data/val/images',
        'data/val/labels',
        'data/test/images',
        'data/test/labels',
        'logging/project_init',
        'logging/data_conversion',
        'logging/train',
        'logging/val',
        'logging/infer',
        'logging/performance_test',
        'logging/test_log',
        'runs/detect',
        'runs/val',
        'runs/infer',
        'models/pretrained',
        'models/checkpoints',
        'output',
        'temp'
    ]
    
    created_count = 0
    existed_count = 0
    problem_count = 0
    
    for directory in directories:
        try:
            full_path = Path(base_path) / directory
            if full_path.exists():
                logging.info(f"✅ 目录已存在: {directory}")
                existed_count += 1
            else:
                full_path.mkdir(parents=True, exist_ok=True)
                logging.info(f"🆕 创建新目录: {directory}")
                created_count += 1
        except Exception as e:
            logging.error(f"❌ 创建目录失败: {directory} - 错误: {str(e)}")
            problem_count += 1
    
    logging.info(f"📊 目录创建完成 - 新建: {created_count}, 已存在: {existed_count}, 问题: {problem_count}")
    return created_count, existed_count, problem_count


@time_it
def check_system_requirements():
    """检查系统要求"""
    logging.info("🔍 检查系统要求...")
    
    # 检查Python版本
    import sys
    python_version = sys.version_info
    logging.info(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查可用磁盘空间
    try:
        import shutil
        total, used, free = shutil.disk_usage(Path.cwd())
        free_gb = free // (1024**3)
        logging.info(f"可用磁盘空间: {free_gb} GB")
        
        if free_gb < 5:
            logging.warning("⚠️ 磁盘空间不足5GB，建议释放更多空间")
    except Exception as e:
        logging.error(f"无法检查磁盘空间: {str(e)}")


@time_it
def create_sample_configs():
    """创建示例配置文件"""
    config_dir = Path("configs")
    
    # 创建训练配置示例
    train_config_content = """# YOLOv8 训练配置文件示例
# 请根据你的实际需求修改以下参数

# 模型配置
model: yolov8n.pt  # 预训练模型路径

# 数据配置
data: configs/data.yaml  # 数据集配置文件路径

# 训练参数
epochs: 100
batch: 16
imgsz: 640
device: 0  # GPU设备号，使用CPU请设置为 'cpu'

# 优化器配置
lr0: 0.01
momentum: 0.937
weight_decay: 0.0005

# 数据增强
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
"""
    
    # 创建数据集配置示例
    dataset_config_content = """# 数据集配置文件
# 请根据你的实际数据路径修改

# 数据路径
path: data  # 数据集根目录
train: train/images  # 训练图片路径
val: val/images      # 验证图片路径
test: test/images    # 测试图片路径（可选）

# 类别数量
nc: 1  # 请根据你的实际类别数量修改

# 类别名称
names:
  0: medical_object  # 请根据你的实际类别名称修改
"""
    
    try:
        # 写入训练配置文件
        train_config_path = config_dir / "train_config.yaml"
        if not train_config_path.exists():
            with open(train_config_path, 'w', encoding='utf-8') as f:
                f.write(train_config_content)
            logging.info(f"📝 创建训练配置文件: {train_config_path}")
        else:
            logging.info(f"📝 训练配置文件已存在: {train_config_path}")
        
        # 写入数据集配置文件
        dataset_config_path = config_dir / "data.yaml"
        if not dataset_config_path.exists():
            with open(dataset_config_path, 'w', encoding='utf-8') as f:
                f.write(dataset_config_content)
            logging.info(f"📝 创建数据集配置文件: {dataset_config_path}")
        else:
            logging.info(f"📝 数据集配置文件已存在: {dataset_config_path}")
            
    except Exception as e:
        logging.error(f"❌ 创建配置文件失败: {str(e)}")


@time_it
def print_detailed_user_guide(base_path):
    """打印详细的用户指引"""
    guide_content = f"""
{'='*80}
🎯 MedicalYOLO 项目初始化完成！
{'='*80}

📁 项目目录结构说明：
├── configs/                   # 配置文件目录（训练、数据集等yaml文件）
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据目录（需用户手动添加）
│   │   ├── images/            # 原始医学图像（用户放置）
│   │   └── annotations/       # 原始标注文件（COCO JSON，用户放置）
│   ├── train/                 # 训练集目录（自动生成）
│   │   ├── images/            # 训练图片
│   │   └── labels/            # 训练标签
│   ├── val/                   # 验证集目录（自动生成）
│   │   ├── images/            # 验证图片
│   │   └── labels/            # 验证标签
│   └── test/                  # 测试集目录（自动生成）
│       ├── images/            # 测试图片
│       └── labels/            # 测试标签
├── models/                    # 模型文件目录
│   ├── pretrained/            # 预训练模型（用户可放置下载的.pt文件）
│   └── checkpoints/           # 训练过程中的模型权重保存
├── logging/                   # 日志文件目录
│   ├── project_init/          # 初始化日志
│   ├── data_conversion/       # 数据转换相关日志
│   ├── train/                 # 训练过程日志
│   ├── val/                   # 验证过程日志
│   ├── infer/                 # 推理过程日志
│   ├── performance_test/      # 性能测试日志
│   ├── test_log/              # 测试日志
│   └── general/               # 通用日志
├── runs/                      # YOLO运行结果目录
│   ├── detect/                # 检测结果
│   ├── val/                   # 验证结果
│   └── infer/                 # 推理结果
├── output/                    # 最终输出目录（如可视化结果、导出文件等）
├── temp/                      # 临时文件目录（缓存、临时数据等）
└── utils/                     # 工具模块目录（项目核心工具代码）
    ├── paths.py               # 路径管理模块
    ├── logging_utils.py       # 日志工具模块
    ├── performance_utils.py   # 性能测量模块
    ├── initialize_project.py  # 项目初始化模块
    └── ...                    # 其他工具模块

🚀 接下来你需要完成以下步骤：

1️⃣ 【必需】准备数据集
   📂 将你的医学图像文件放入: {base_path}/data/raw/images/
      - 支持格式: .jpg, .jpeg, .png, .bmp
      - 建议分辨率: 640x640 或更高
      - 文件命名: 使用英文和数字，避免特殊字符
   
   📄 将COCO格式的标注文件放入: {base_path}/data/raw/annotations/
      - 文件格式: .json
      - 必须符合COCO标注格式
      - 标注文件名应与图像对应

2️⃣ 【可选且推荐】下载预训练模型
   🌐 访问 https://github.com/ultralytics/yolov8/releases
   📥 下载适合的预训练模型 (如: yolov8n.pt, yolov8s.pt, yolov8m.pt)
   📁 将模型文件放入: {base_path}/models/pretrained/

3️⃣ 【必需】配置文件设置
   ⚙️ 编辑配置文件: {base_path}/configs/data.yaml
      - 修改类别数量 (nc)
      - 修改类别名称 (names)
      - 确认数据路径正确
   
   ⚙️ 编辑训练配置: {base_path}/configs/train_config.yaml
      - 根据GPU内存调整batch大小
      - 设置训练轮数 (epochs)
      - 选择合适的模型大小

4️⃣ 【重要】环境检查
   🐍 确保Python版本 >= 3.8
   📦 安装必需的包:
      pip install ultralytics
      pip install opencv-python
      pip install pillow
      pip install numpy
      pip install matplotlib

5️⃣ 【验证】数据准备验证
   运行以下命令验证数据准备情况:
   python -c "
import os
print('图像文件数量:', len([f for f in os.listdir('data/raw/images') if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]))
print('标注文件数量:', len([f for f in os.listdir('data/raw/annotations') if f.endswith('.json')]))
"

💡 提示与建议：

🔸 数据质量建议：
  - 图像清晰度要高，避免模糊
  - 标注准确，边界框紧贴目标
  - 数据分布均衡，各类别样本充足
  - 建议训练:验证:测试 = 7:2:1

🔸 训练参数建议：
  - 首次训练使用较小的batch size (8-16)
  - 学习率从0.01开始，根据效果调整
  - epochs建议从100开始，观察收敛情况

🔸 硬件建议：
  - GPU内存 >= 6GB (推荐8GB+)
  - CPU内存 >= 16GB
  - 可用磁盘空间 >= 10GB

📞 需要帮助？
  - 查看项目文档和示例
  - 检查日志文件了解详细信息
  - 确保所有依赖包正确安装

✅ 准备完成后，你就可以开始训练你的医学目标检测模型了！

{'='*80}
"""
    
    print(guide_content)
    logging.info("📋 用户指引已显示")


@time_it
def check_data_status(base_path):
    """检查数据目录状态"""
    raw_images_dir = base_path / 'data/raw/images'
    raw_annotations_dir = base_path / 'data/raw/annotations'
    
    # 检查图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_count = 0
    if raw_images_dir.exists():
        image_files = [f for f in raw_images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        image_count = len(image_files)
        
        if image_count > 0:
            # 检查图像尺寸分布
            try:
                from PIL import Image
                sizes = []
                for img_file in image_files[:10]:  # 只检查前10张
                    try:
                        with Image.open(img_file) as img:
                            sizes.append(img.size)
                    except:
                        continue
                
                if sizes:
                    avg_width = sum(s[0] for s in sizes) / len(sizes)
                    avg_height = sum(s[1] for s in sizes) / len(sizes)
                    logging.info(f"📏 图像平均尺寸: {avg_width:.0f}x{avg_height:.0f}")
            except ImportError:
                logging.warning("⚠️ 未安装PIL，无法检查图像尺寸")
    
    # 检查标注文件
    annotation_count = 0
    if raw_annotations_dir.exists():
        json_files = [f for f in raw_annotations_dir.iterdir() 
                     if f.suffix.lower() == '.json']
        annotation_count = len(json_files)
    
    logging.info(f"📊 数据状态检查:")
    logging.info(f"   - 图像文件数量: {image_count}")
    logging.info(f"   - 标注文件数量: {annotation_count}")
    
    # 提供数据状态建议
    if image_count == 0:
        logging.warning("⚠️ 未发现图像文件，请确保图像放在正确位置")
    elif image_count < 100:
        logging.warning("⚠️ 图像数量较少，建议至少100张用于训练")
    else:
        logging.info("✅ 图像数量充足")
    
    if annotation_count == 0:
        logging.warning("⚠️ 未发现标注文件，请确保标注文件放在正确位置")
    else:
        logging.info("✅ 发现标注文件")
    
    return image_count, annotation_count


@time_it
def initialize_project():
    """主初始化函数"""
    start_time = time.perf_counter()
    
    # 使用当前目录作为项目根目录
    base_path = Path(__file__).parent.parent
    log_file_path = setup_logging(base_path)
    
    logging.info("🚀 开始初始化MedicalYOLO项目")
    logging.info(f"📁 项目根目录: {base_path.absolute()}")
    
    # 检查系统要求
    check_system_requirements()
    
    # 创建目录结构
    created, existed, problems = create_directory_structure(base_path)
    
    # 创建示例配置文件
    create_sample_configs()
    
    # 检查数据状态
    image_count, annotation_count = check_data_status(base_path)
    
    # 计算总执行时间
    total_time = time.perf_counter() - start_time
    if total_time >= 1.0:
        time_str = f"{total_time:.4f} 秒"
    else:
        time_str = f"{total_time * 1000:.4f} 毫秒"
    
    logging.info(f"⏱️ 初始化总耗时: {time_str}")
    logging.info("✅ MedicalYOLO项目初始化完成")
    
    # 显示详细的用户指引
    print_detailed_user_guide(base_path)
    
    # 最终状态报告
    print(f"\n📋 初始化完成报告:")
    print(f"   ✅ 新建目录: {created} 个")
    print(f"   📁 已存在目录: {existed} 个") 
    print(f"   ❌ 问题目录: {problems} 个")
    print(f"   🖼️ 当前图像文件: {image_count} 个")
    print(f"   📄 当前标注文件: {annotation_count} 个")
    print(f"   ⏱️ 总执行时间: {time_str}")
    print(f"   📝 详细日志: {log_file_path}")
    
    # 数据准备状态提示
    if image_count == 0 or annotation_count == 0:
        print(f"\n⚠️  请按照上述指引准备数据文件后再开始训练！")
    else:
        print(f"\n🎉 数据文件已就绪，可以开始配置和训练！")


if __name__ == "__main__":
    initialize_project()