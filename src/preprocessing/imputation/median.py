import pandas as pd
import numpy as np

def replace_missing_values(df):
    numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                          'BMI', 'DiabetesPedigreeFunction', 'Age']
    # Check
    missing_data = df[numerical_features].isnull().sum()
    print("-------Dữ liệu bị thiếu:")
    print(missing_data)
    # Replace
    df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())

    # Including Zero values
    # Check
    cols = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]
    for col in cols:
        zero_values = len(df[df[col] <= 0])
        print("-------Các cột có dữ liệu bằng 0 hoặc dưới 0 của thuộc tính {} là {}".format(col,zero_values))

    # Replace
    for col in cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)
        df[col] = np.where(df[col] <= 0, median, df[col]) # condition true => median else stay df[col]
    print("-------Điền giá trị thiếu hoàn tất!")
    return df