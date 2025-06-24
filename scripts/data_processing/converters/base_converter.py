from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class BaseConverter(ABC):
    """转换器基类"""
    
    def __init__(self):
        self.class_names = []
    
    @abstractmethod
    def convert(self, input_path: str, output_path: str, class_mapping: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        执行转换
        
        Args:
            input_path: 输入路径
            output_path: 输出路径
            class_mapping: 类别映射字典
            
        Returns:
            转换结果信息
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_path: str) -> bool:
        """验证输入格式是否正确"""
        pass