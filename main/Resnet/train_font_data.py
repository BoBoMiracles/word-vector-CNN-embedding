#!/usr/bin/env pyhon3
# -*- coding:utf-8 -*-
"""
Created on 2024-03-23 7:14 PM
@file: train_font_data.py
@author: bobo-Miracles
"""
import os
import numpy as np
from keras.optimizers import Adam
from keras.preprocessing import image
from fontTools.ttLib import TTFont
from resnet50 import createcnn
from PIL import Image, ImageFont, ImageDraw
from keras.utils import img_to_array
import unicodedata
import pickle


def preprocess_image(img):

    # 调整图像大小为模型期望的输入尺寸（224x224）
    img = img.resize((224, 224))

    # 将图像转换为数组
    img_array = img_to_array(img)
    img_array = np.array(img_array)

    # 将像素值归一化到 [0, 1] 范围内
    img_array /= 255.0

    # 对图像进行零填充，使其尺寸与模型输入的尺寸相匹配
    # 这里不需要进行零填充，因为已经在模型中进行了零填充操作

    return img_array

# 设置路径
# font_file_path = 'path/to/your/font.otf'  # 替换为你的 OTF 文件路径
# model_save_path = 'path/to/save/your/model.h5'  # 替换为模型保存路径

# 加载 ttf 文件
font = ImageFont.truetype('Noto_Sans_SC/NotoSansSC-VariableFont_wght.ttf', 24)
# 获取ttf文件中的所有字符
characters = []

# 遍历Unicode字符范围
for i in range(0x4e00, 0x9fa6):  # 汉字的Unicode范围
    try:
        char = chr(i)
        # width, _ = font.getsize(char)  # 检查字符是否在字体中
        characters.append(char)
    except Exception as e:
        print("Error for character", char, ":", e)

# 输出所有字符
# print(characters)

# 创建一个空数组来存储每个汉字的向量表示
vector_char_map = {}

# 加载 ResNet50 模型
model = createcnn()

# 编译模型
model.compile(optimizer=Adam(), loss='mean_squared_error')

# 循环遍历每个字符

# 获取字符的字形数据
# print(characters)
glyph = font.getmask(characters[1])

# 获取字形的边界框
xmin, ymin, xmax, ymax = glyph.getbbox()

# 计算字形的宽度和高度
width = xmax - xmin
height = ymax - ymin

# 计算字形在图像中的位置
x_offset = (224 - width) // 2 - xmin  # 考虑x轴上的偏移量
y_offset = (224 - height) // 2 - ymin  # 考虑y轴上的偏移量

# 设置字体颜色为黑色
font_color = (0, 0, 0)

for char in characters:
    # 创建一个图像对象，用于绘制当前字符
    img = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(img)


    # 绘制字形到图像上
    draw.text((x_offset, y_offset), char, font=font, fill=font_color)

    # img.show()

    # 预处理图像（与模型的预处理一致）
    img_array = preprocess_image(img)

    # 提取特征向量
    vector = model.predict(np.expand_dims(img_array, axis=0))

    # 添加特征向量到字典中，以字符为键，特征向量为值
    vector_char_map[char] = vector
    # print(a)

# print(vector_char_map)

# 将结果保存到文件中或进行其他操作
# 文件路径
file_path = "vector_char_map.pkl"

# 将vector_char_map保存到文件中
with open(file_path, 'wb') as f:
    pickle.dump(vector_char_map, f)

print("保存成功：", file_path)

# # 从文件中加载vector_char_map
# with open(file_path, 'rb') as f:
#     loaded_vector_char_map = pickle.load(f)
#
# print("加载成功：", loaded_vector_char_map)