#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :performance_utils.py
# @Time      :2025/6/23 14:35:00
# @Author    :雨霓同学
# @Project   :MedicalYOLO
# @Function  :性能测量工具

import time
import functools
import logging
from typing import Callable, Any, Optional, Tuple


def format_time(seconds: float) -> Tuple[float, str]:
    """
    自动选择合适的时间单位
    :param seconds: 秒数
    :return: (格式化后的数值, 单位)
    """
    if seconds >= 1.0:
        return seconds, "秒"
    elif seconds >= 0.001:
        return seconds * 1000, "毫秒"
    elif seconds >= 0.000001:
        return seconds * 1000000, "微秒"
    else:
        return seconds * 1000000000, "纳秒"


def time_it(func: Callable = None, *,
            repeat_times: int = 1,
            logger_instance: Optional[logging.Logger] = None,
            show_args: bool = True,
            precision: int = 4) -> Callable:
    """
    性能测量装饰器 - 支持不修改源码的情况下测量函数执行时间
    
    :param func: 被装饰的函数
    :param repeat_times: 重复执行次数，用于计算平均时间
    :param logger_instance: 日志记录器实例
    :param show_args: 是否在日志中显示函数参数
    :param precision: 时间显示精度
    :return: 装饰后的函数
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            func_name = f.__name__
            
            # 构建参数信息
            args_info = ""
            if show_args and (args or kwargs):
                args_str = ", ".join(repr(arg) for arg in args)
                kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
                all_args = [s for s in [args_str, kwargs_str] if s]
                args_info = f"({', '.join(all_args)})"
            
            # 记录开始时间
            start_time = time.perf_counter()
            
            # 执行函数指定次数
            result = None
            for i in range(repeat_times):
                if i == repeat_times - 1:  # 最后一次执行保存结果
                    result = f(*args, **kwargs)
                else:
                    f(*args, **kwargs)
            
            # 计算执行时间
            end_time = time.perf_counter()
            total_time = end_time - start_time
            avg_time = total_time / repeat_times
            
            # 统一时间单位显示 - 使用更小的单位保持一致性
            if total_time >= 1.0:
                # 如果总时间超过1秒，都用秒显示
                formatted_total_time, total_unit = total_time, "秒"
                formatted_avg_time, avg_unit = avg_time, "秒"
            elif total_time >= 0.001:
                # 如果总时间超过1毫秒，都用毫秒显示
                formatted_total_time, total_unit = total_time * 1000, "毫秒"
                formatted_avg_time, avg_unit = avg_time * 1000, "毫秒"
            else:
                # 否则都用微秒显示
                formatted_total_time, total_unit = total_time * 1000000, "微秒"
                formatted_avg_time, avg_unit = avg_time * 1000000, "微秒"
            
            # 构建日志消息
            if repeat_times == 1:
                message = (f"⏱️ 函数 '{func_name}{args_info}' "
                          f"执行时间: {formatted_avg_time:.{precision}f} {avg_unit}")
            else:
                message = (f"⏱️ 函数 '{func_name}{args_info}' "
                          f"执行 {repeat_times} 次, "
                          f"总时间: {formatted_total_time:.{precision}f} {total_unit}, "
                          f"平均时间: {formatted_avg_time:.{precision}f} {avg_unit}")
            
            # 输出到日志或控制台
            if logger_instance:
                logger_instance.info(message)
            else:
                print(f"[TIME_IT] {message}")
                
            return result
        return wrapper
    
    # 支持带参数和不带参数的装饰器调用
    if func is None:
        return decorator
    else:
        return decorator(func)


def measure_execution_time(func: Callable, 
                          args: tuple = (), 
                          kwargs: dict = None,
                          repeat_times: int = 1,
                          logger_instance: Optional[logging.Logger] = None,
                          precision: int = 4) -> Tuple[Any, float]:
    """
    手动测量函数执行时间的工具函数
    
    :param func: 要测量的函数
    :param args: 函数的位置参数
    :param kwargs: 函数的关键字参数
    :param repeat_times: 重复执行次数
    :param logger_instance: 日志记录器实例
    :param precision: 时间显示精度
    :return: (函数执行结果, 平均执行时间)
    """
    if kwargs is None:
        kwargs = {}
    
    func_name = func.__name__
    
    # 记录开始时间
    start_time = time.perf_counter()
    
    # 执行函数指定次数
    result = None
    for i in range(repeat_times):
        if i == repeat_times - 1:  # 最后一次执行保存结果
            result = func(*args, **kwargs)
        else:
            func(*args, **kwargs)
    
    # 计算执行时间
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time = total_time / repeat_times
    
    # 统一时间单位显示
    if total_time >= 1.0:
        formatted_total_time, total_unit = total_time, "秒"
        formatted_avg_time, avg_unit = avg_time, "秒"
    elif total_time >= 0.001:
        formatted_total_time, total_unit = total_time * 1000, "毫秒"
        formatted_avg_time, avg_unit = avg_time * 1000, "毫秒"
    else:
        formatted_total_time, total_unit = total_time * 1000000, "微秒"
        formatted_avg_time, avg_unit = avg_time * 1000000, "微秒"
    
    # 构建日志消息
    if repeat_times == 1:
        message = (f"⏱️ 手动测量函数 '{func_name}' "
                  f"执行时间: {formatted_avg_time:.{precision}f} {avg_unit}")
    else:
        message = (f"⏱️ 手动测量函数 '{func_name}' "
                  f"执行 {repeat_times} 次, "
                  f"总时间: {formatted_total_time:.{precision}f} {total_unit}, "
                  f"平均时间: {formatted_avg_time:.{precision}f} {avg_unit}")
    
    # 输出到日志或控制台
    if logger_instance:
        logger_instance.info(message)
    else:
        print(f"[TIME_IT] {message}")
    
    return result, avg_time


class PerformanceProfiler:
    """性能分析器 - 支持批量测量和统计"""
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        self.logger = logger_instance
        self.results = []
    
    def profile_function(self, func: Callable, 
                        args: tuple = (), 
                        kwargs: dict = None,
                        repeat_times: int = 1,
                        label: str = None) -> float:
        """
        分析单个函数性能
        
        :param func: 要分析的函数
        :param args: 函数参数
        :param kwargs: 函数关键字参数
        :param repeat_times: 重复次数
        :param label: 自定义标签
        :return: 平均执行时间
        """
        if kwargs is None:
            kwargs = {}
        
        func_label = label or func.__name__
        result, avg_time = measure_execution_time(
            func, args, kwargs, repeat_times, self.logger
        )
        
        self.results.append({
            'label': func_label,
            'avg_time': avg_time,
            'repeat_times': repeat_times
        })
        
        return avg_time
    
    def get_summary(self) -> str:
        """获取性能分析摘要"""
        if not self.results:
            return "暂无性能分析数据"
        
        summary_lines = ["=" * 60, "性能分析摘要 (按平均执行时间排序)", "=" * 60]
        
        # 按平均时间排序
        sorted_results = sorted(self.results, key=lambda x: x['avg_time'], reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            formatted_time, unit = format_time(result['avg_time'])
            summary_lines.append(
                f"{i:2d}. {result['label']:<30} "
                f"{formatted_time:>8.4f} {unit:<4} "
                f"(重复{result['repeat_times']}次)"
            )
        
        summary_lines.append("=" * 60)
        return "\n".join(summary_lines)
    
    def print_summary(self):
        """打印性能分析摘要"""
        summary = self.get_summary()
        if self.logger:
            for line in summary.split('\n'):
                self.logger.info(line)
        else:
            print(summary)


# 使用示例和测试代码
if __name__ == "__main__":
    import math
    import sys
    from pathlib import Path
    
    # 添加项目根目录到Python路径
    sys.path.append(str(Path(__file__).parent.parent))
    
    from utils.paths import LOGS_DIR
    from utils.logging_utils import setup_logger
    
    # 设置日志记录器
    logger = setup_logger(
        base_path=LOGS_DIR,
        log_type="performance_test",
        logger_name="PerformanceTest"
    )
    
    # 示例1: 装饰器使用 - 快速函数
    @time_it(repeat_times=100, logger_instance=logger)
    def fast_function(n):
        return sum(range(n))
    
    # 示例2: 装饰器使用 - 慢速函数
    @time_it(logger_instance=logger)
    def slow_function():
        time.sleep(0.1)
        return "完成"
    
    # 示例3: 手动测量
    def math_operations(n):
        result = 0
        for i in range(n):
            result += math.sqrt(i + 1)
        return result
    
    # 测试函数
    logger.info("开始性能测试")
    
    # 测试装饰器
    fast_result = fast_function(1000)
    slow_result = slow_function()
    
    # 测试手动测量
    manual_result, manual_time = measure_execution_time(
        math_operations, 
        args=(100,), 
        repeat_times=5, 
        logger_instance=logger
    )
    
    # 测试性能分析器
    profiler = PerformanceProfiler(logger)
    profiler.profile_function(fast_function, args=(1000,), repeat_times=100, label="快速求和")
    profiler.profile_function(math_operations, args=(5000,), repeat_times=3, label="数学运算")
    profiler.profile_function(slow_function, repeat_times=2, label="延时函数")
    
    profiler.print_summary()
    
    logger.info("性能测试完成")