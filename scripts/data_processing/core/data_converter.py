"""
统一数据转换接口
支持多种格式之间的转换
"""
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from ..converters import (
    BaseConverter, 
    CocoToYoloConverter
)

class DataConverter:
    """统一数据转换器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._converters = {}
        self._register_converters()
    
    def _register_converters(self):
        """注册所有可用的转换器"""
        self._converters = {
            'coco_to_yolo': CocoToYoloConverter(),
            # 将来可以添加更多转换器
            # 'pascal_to_yolo': PascalVocToYoloConverter(),
            # 'labelme_to_yolo': LabelmeToYoloConverter(),
        }
    
    def get_available_conversions(self) -> list:
        """获取所有可用的转换类型"""
        return list(self._converters.keys())
    
    def convert(self, 
                conversion_type: str,
                input_path: str, 
                output_path: str,
                class_mapping: Optional[Dict[str, int]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        执行数据转换
        
        Args:
            conversion_type: 转换类型 (如 'coco_to_yolo')
            input_path: 输入路径
            output_path: 输出路径
            class_mapping: 类别映射
            **kwargs: 其他参数
            
        Returns:
            转换结果
        """
        if conversion_type not in self._converters:
            raise ValueError(f"不支持的转换类型: {conversion_type}")
        
        converter = self._converters[conversion_type]
        self.logger.info(f"开始执行 {conversion_type} 转换")
        
        try:
            result = converter.convert(input_path, output_path, class_mapping, **kwargs)
            self.logger.info(f"转换完成: {conversion_type}")
            return result
        except Exception as e:
            self.logger.error(f"转换失败 {conversion_type}: {e}")
            raise
    
    def validate_input(self, conversion_type: str, input_path: str) -> bool:
        """验证输入格式"""
        if conversion_type not in self._converters:
            return False
        
        converter = self._converters[conversion_type]
        return converter.validate_input(input_path)
