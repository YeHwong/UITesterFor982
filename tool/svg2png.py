#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023-08-03 14:38
# @File: svg2png.py
# @Author: YeHwong
# @Email: 598318610@qq.com
# @Version ：1.0.0
import os
import fnmatch
import cairosvg

def transition(files):
    for file_item in files:
        directory = r'C:\Users\Administrator\Desktop\UI图标\Basic UI Icons\SVGs'
        png_dir = r'C:\Users\Administrator\Desktop\UI图标\Basic UI Icons\pngs'
        svg_path = f'{directory}\\{file_item}'
        png_path = f'{png_dir}\\{file_item}'
        cairosvg.svg2png(url=svg_path, write_to=png_path, dpi=600)

def search_files():
    directory = r'C:\Users\Administrator\Desktop\UI图标\Basic UI Icons\SVGs'
    pattern = '*.svg'
    result = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            result.append(filename)
    if result:
        print(result)
    else:
        print('File not found.')
    return result


if __name__ == '__main__':
    files = search_files()
    transition(files)
