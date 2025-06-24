import os
import shutil
from pathlib import Path
from typing import List
import logging

def move_files_by_extension(source_dir: str, target_dir: str, extensions: List[str]) -> int:
    """
    根据文件扩展名移动文件
    
    Args:
        source_dir: 源目录
        target_dir: 目标目录
        extensions: 要移动的文件扩展名列表
        
    Returns:
        移动的文件数量
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    logger = logging.getLogger(__name__)
    
    for ext in extensions:
        pattern = f"*.{ext.lstrip('.')}"
        for file_path in source_path.glob(pattern):
            try:
                target_file = target_path / file_path.name
                shutil.move(str(file_path), str(target_file))
                moved_count += 1
                logger.debug(f"移动文件: {file_path} -> {target_file}")
            except Exception as e:
                logger.error(f"移动文件失败 {file_path}: {e}")
    
    return moved_count

def cleanup_temp_directory(temp_dir: str) -> bool:
    """
    清理临时目录
    
    Args:
        temp_dir: 临时目录路径
        
    Returns:
        是否成功清理
    """
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            return True
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"清理临时目录失败 {temp_dir}: {e}")
        return False