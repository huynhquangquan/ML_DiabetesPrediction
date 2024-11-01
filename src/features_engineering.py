import pandas as pd
import numpy as np
from pathlib import Path

def check_correlation(dataset, threshold, cat_feature):
    dataset = dataset.drop(cat_feature,axis=1)
    col_corr = set()  # Tập hợp các thuộc tính đã bị xóa
    corr_matrix = dataset.corr(numeric_only=True) # Chỉ lấy thuộc tính dạng numeric
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]  # Lấy tên cột bị xóa
                col_corr.add(colname)
    return col_corr

def remove_features(dataset, features):
    print("Thuộc tính loại bỏ: ")
    print(features)
    return dataset.drop(features,axis=1)