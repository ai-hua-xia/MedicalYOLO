import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import datetime
from .base_converter import BaseConverter

class CocoToYoloConverter(BaseConverter):
    """COCO JSON格式转YOLO TXT格式转换器"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def validate_input(self, input_path: str) -> bool:
        if not os.path.exists(input_path):
            print(f"❌ 输入路径不存在: {input_path}")
            self.logger.error(f"输入路径不存在: {input_path}")
            return False
        
        json_files = list(Path(input_path).glob("*.json"))
        if not json_files:
            print(f"❌ 未找到JSON文件: {input_path}")
            self.logger.error(f"未找到JSON文件: {input_path}")
            return False
        
        print(f"✅ 找到 {len(json_files)} 个JSON文件:")
        for json_file in json_files:
            print(f"   - {json_file.name}")
        
        return True
    
    def _load_coco_annotations(self, json_files: List[Path]) -> Dict[str, Any]:
        print(f"📂 开始加载和合并COCO数据...")
        merged_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        category_id_mapping = {}
        next_category_id = 0
        for i, json_file in enumerate(json_files, 1):
            print(f"   [{i}/{len(json_files)}] 处理文件: {json_file.name}")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"      - 图像数量: {len(data.get('images', []))}")
                print(f"      - 标注数量: {len(data.get('annotations', []))}")
                print(f"      - 类别数量: {len(data.get('categories', []))}")
                for category in data.get('categories', []):
                    cat_name = category['name']
                    if cat_name not in category_id_mapping:
                        category_id_mapping[cat_name] = next_category_id
                        merged_data['categories'].append({
                            'id': next_category_id,
                            'name': cat_name
                        })
                        print(f"      - 新类别: {cat_name} (ID: {next_category_id})")
                        next_category_id += 1
                merged_data['images'].extend(data.get('images', []))
                for ann in data.get('annotations', []):
                    original_cat_id = ann['category_id']
                    cat_name = None
                    for cat in data.get('categories', []):
                        if cat['id'] == original_cat_id:
                            cat_name = cat['name']
                            break
                    if cat_name:
                        ann['category_id'] = category_id_mapping[cat_name]
                        merged_data['annotations'].append(ann)
            except Exception as e:
                print(f"❌ 加载JSON文件失败 {json_file}: {e}")
                self.logger.error(f"加载JSON文件失败 {json_file}: {e}")
                raise
        print(f"✅ 数据合并完成:")
        print(f"   - 总图像数: {len(merged_data['images'])}")
        print(f"   - 总标注数: {len(merged_data['annotations'])}")
        print(f"   - 总类别数: {len(merged_data['categories'])}")
        print(f"   - 类别列表: {[cat['name'] for cat in merged_data['categories']]}")
        return merged_data
    
    def _convert_bbox_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        x, y, w, h = bbox
        x_center = (x + w / 2.0) / img_width
        y_center = (y + h / 2.0) / img_height
        width = w / img_width
        height = h / img_height
        return [x_center, y_center, width, height]
    
    def convert(self, input_path: str, output_path: str, class_mapping: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        print(f"🚀 开始COCO到YOLO转换...")
        if not self.validate_input(input_path):
            raise ValueError(f"输入验证失败: {input_path}")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建输出目录: {output_path}")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = output_path / f"temp_{timestamp}"
        temp_dir.mkdir(exist_ok=True)
        print(f"📁 创建临时目录: {temp_dir}")
        try:
            json_files = list(Path(input_path).glob("*.json"))
            self.logger.info(f"找到 {len(json_files)} 个JSON文件")
            coco_data = self._load_coco_annotations(json_files)
            image_info = {img['id']: img for img in coco_data['images']}
            print(f"📊 创建图像映射表，共 {len(image_info)} 个图像")
            if class_mapping:
                category_mapping = class_mapping
                print(f"🏷️  使用自定义类别映射: {category_mapping}")
            else:
                category_mapping = {cat['name']: cat['id'] for cat in coco_data['categories']}
                print(f"🏷️  使用自动类别映射: {category_mapping}")
            self.class_names = list(category_mapping.keys())
            print(f"🏷️  类别顺序: {self.class_names}")
            print(f"🔄 开始转换标注...")
            converted_files = []
            converted_annotations = 0
            for i, annotation in enumerate(coco_data['annotations'], 1):
                if i % 100 == 0 or i == len(coco_data['annotations']):
                    print(f"   进度: [{i}/{len(coco_data['annotations'])}] 标注已处理")
                image_id = annotation['image_id']
                if image_id not in image_info:
                    continue
                img_info = image_info[image_id]
                img_filename = img_info['file_name']
                img_width = img_info['width']
                img_height = img_info['height']
                txt_filename = Path(img_filename).stem + '.txt'
                txt_path = temp_dir / txt_filename
                bbox = annotation['bbox']
                yolo_bbox = self._convert_bbox_to_yolo(bbox, img_width, img_height)
                category_id = annotation['category_id']
                with open(txt_path, 'a', encoding='utf-8') as f:
                    line = f"{category_id} {' '.join(map(str, yolo_bbox))}\n"
                    f.write(line)
                if txt_filename not in converted_files:
                    converted_files.append(txt_filename)
                converted_annotations += 1
            # 保存类别名称文件
            classes_file = output_path / 'classes.txt'
            with open(classes_file, 'w', encoding='utf-8') as f:
                for class_name in self.class_names:
                    f.write(f"{class_name}\n")
            print(f"💾 保存类别文件: {classes_file}")
            result = {
                'temp_dir': str(temp_dir),
                'converted_files': converted_files,
                'class_names': self.class_names,
                'total_annotations': len(coco_data['annotations']),
                'total_images': len(set(ann['image_id'] for ann in coco_data['annotations'])),
                'converted_annotations': converted_annotations,
                'converted_label_files': len(converted_files)
            }
            print(f"✅ 转换完成!")
            return result
        except Exception as e:
            print(f"❌ 转换过程中发生错误: {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                print(f"🧹 已清理临时目录: {temp_dir}")
            raise e