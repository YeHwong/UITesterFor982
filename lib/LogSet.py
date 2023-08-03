#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023-02-08 9:37
# @File: LogSet.py
# @Author: YeHwong
# @Email: 598318610@qq.com
# @Version ：2.0.0

import logging
import os
import datetime


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,  # debug级别 调试过程中使用DEBUG等级，如算法中每个循环的中间状态
        'info': logging.INFO,  # info 信息级别   处理请求或者状态变化等日常事务
        'warning': logging.WARNING,  # 警告级别     发生很重要的事件，但是并不是错误时，如用户登录密码错误
        'error': logging.ERROR,  # 错误级别     发生错误时，如IO操作失败或者连接问题
        'critical': logging.CRITICAL,  # 严重级别     特别糟糕的事情，如内存耗尽、磁盘空间为空，一般很少使用
        'fatal': logging.FATAL  # 致命错误
    }  # 日志级别关系映射

    def __init__(self, base_dir="../APP_LOG/", level='info',
                 fmt='%(levelname)s :%(asctime)s - [%(filename)s -->line:%(lineno)d]: %(message)s'):
        """
            Log打印输出类

        :param base_dir: 保存LOG的文件夹路径
        :type base_dir: str
        :param level: log 等级 <debug info warning error critical fatal>
        :type level: str
        :param fmt: 日志格式
        :type fmt: str
        """
        clean_empty(base_dir)
        base_dir = base_dir  # 设置Log保存的文件夹

        datetime_now = datetime.datetime.now().strftime('%Y-%m-%d+%H_%M_%S').split('+')
        date_now = datetime_now[0]  # 当前日期
        # time_now = datetime_now[1]  # 当前时间

        log_dir_dic = {'log_debug_dir': f'{base_dir}/debug/',
                       'log_info_dir': f'{base_dir}/info/',
                       'log_warning_dir': f'{base_dir}/warning/',
                       'log_error_dir': f'{base_dir}/error/'
                       }
        for log_dir in log_dir_dic:
            if not os.path.exists(log_dir_dic[log_dir]):  # 检查文件夹是否存在
                os.makedirs(log_dir_dic[log_dir], exist_ok=True)  # 新建文件夹，设置存在标志

        debug_file_handle = logging.FileHandler(f"{log_dir_dic['log_debug_dir']}{date_now}.log", encoding='utf-8')
        debug_file_handle.setLevel("DEBUG")  # 设置输出的日志等级为DEBUG
        info_file_handle = logging.FileHandler(f"{log_dir_dic['log_info_dir']}{date_now}.log", encoding='utf-8')
        info_file_handle.setLevel("INFO")  # 设置输出的日志等级为INFO以上
        warning_file_handle = logging.FileHandler(f"{log_dir_dic['log_warning_dir']}{date_now}.log", encoding='utf-8')
        warning_file_handle.setLevel("WARNING")  # 设置输出的日志等级为WARNING以上
        error_file_handle = logging.FileHandler(f"{log_dir_dic['log_error_dir']}{date_now}.log", encoding='utf8')
        error_file_handle.setLevel("ERROR")  # 设置输出的日志等级为ERROR以上

        self.my_logger = logging.getLogger()  # 创建日志收集器
        self.my_logger.setLevel(self.level_relations.get(level))  # 设置收集器日志级别
        # 设置日志输出的格式
        # 可以通过logging.Formatter指定日志的输出格式，这个参数可以输出很多有用的信息，如下：
        # % (name)s: 收集器名称
        # % (levelno)s: 打印日志级别的数值
        # % (levelname)s: 打印日志级别名称
        # % (pathname)s: 打印当前执行程序的路径，其实就是sys.argv()
        # % (filename)s: 打印当前执行程序名
        # % (funcName)s: 打印日志的当前函数
        # % (lineno)d: 打印日志的当前行号
        # % (asctime)s: 打印日志的时间
        # % (thread) d: 打印线程ID
        # % (threadName)s: 打印线程名称
        # % (process) d: 打印进程ID
        # % (message) s: 打印日志信息
        # ft = "%(asctime)s - [%(filename)s -->line:%(lineno)d] - %(levelname)s: %(message)s"  # 工作中常用的日志格式
        format_str = logging.Formatter(fmt)  # 设置日志格式

        sh = logging.StreamHandler()  # 往屏幕上输出   <---流处理器---->
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        debug_file_handle.setFormatter(format_str)
        info_file_handle.setFormatter(format_str)
        warning_file_handle.setFormatter(format_str)
        error_file_handle.setFormatter(format_str)

        self.my_logger.addHandler(sh)  # 把屏幕输出对象加到logger里
        # 将日志输出渠道添加到日志收集器中
        self.my_logger.addHandler(debug_file_handle)
        self.my_logger.addHandler(info_file_handle)
        self.my_logger.addHandler(warning_file_handle)
        self.my_logger.addHandler(error_file_handle)


# def new_path(path):
#     this_path = Path.cwd()
#     File_Path = Path(f"{this_path}\\{path}\\")  # 获取到当前文件的目录，并检查是否有data文件夹，如果不存在则自动新建data文件
#     if not File_Path.exists():
#         File_Path.mkdir(parents=True, exist_ok=True)


def clean_empty(path):
    """
    遍历文件下所有子文件夹以及子文件，清理空文件夹和空文件
    path:文件路径
    """
    try:
        for (dir_path, dir_names, filenames) in os.walk(path):
            for filename in filenames:
                file_folder = dir_path + '/' + filename
                if os.path.isdir(file_folder):
                    if not os.listdir(file_folder):
                        os.rmdir(dir_path+dir_names)
                elif os.path.isfile(file_folder):
                    if os.path.getsize(file_folder) == 0:
                        os.remove(file_folder)
    except IOError as e:
        print(e)


if __name__ == '__main__':
    my_log = Logger(level="debug").my_logger  # 如果实例化多个Logger对象可能会打印多次。!!!

    my_log.info("error")
