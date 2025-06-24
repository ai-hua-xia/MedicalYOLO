import json
from pathlib import Path

def coco_to_yolo(coco_json_path, output_dir):
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    categories = {cat['id']: cat for cat in coco['categories']}
    annotations = coco['annotations']

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按图片分组标注
    img_to_anns = {}
    for ann in annotations:
        img_to_anns.setdefault(ann['image_id'], []).append(ann)

    for img_id, img in images.items():
        img_w, img_h = img['width'], img['height']
        img_name = Path(img['file_name']).stem
        txt_path = output_dir / f"{img_name}.txt"
        lines = []
        for ann in img_to_anns.get(img_id, []):
            cat_id = ann['category_id']
            # COCO bbox: [x_min, y_min, width, height]
            x, y, w, h = ann['bbox']
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            # YOLO类别id通常从0开始
            yolo_cat_id = cat_id - 1
            lines.append(f"{yolo_cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    print(f"已完成转换，YOLO标签保存在: {output_dir}")

if __name__ == "__main__":
    coco_json = "_annotations.coco_1.json"
    yolo_label_dir = "yolo_txt"
    coco_to_yolo(coco_json, yolo_label_dir)