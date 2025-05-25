#!/usr/bin/env pyhon3
# -*- coding:utf-8 -*-
"""
Created on 2024-03-23 7:49 PM
@file: load_vector.py
@author: bobo-Miracles
"""
import pandas as pd
import pickle
file_path = "vector_char_map.pkl"
# 从文件中加载vector_char_map
# with open(file_path, 'rb') as f:
#     loaded_vector_char_map = pickle.load(f)
# print("加载成功：", loaded_vector_char_map["你"])

data_dict = pd.read_pickle(file_path)
with open("output_CNN_vector.txt", "w", encoding='utf-8') as f:
    for key, value in data_dict.items():
        f.write(f"{key} {value.flatten().tolist()}\n")
# data_df = pd.DataFrame.from_dict(data_dict, orient='index')
# data_df.to_csv("output_CNN_vector.txt", sep=" ", index=False, header=False)
