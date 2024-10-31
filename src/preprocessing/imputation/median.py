import pandas as pd
import numpy as np

def replace_missing_values(train, test):
    trainCopy = train.copy()
    numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                          'BMI', 'DiabetesPedigreeFunction', 'Age']
    # Check
    missing_data_train = train[numerical_features].isnull().sum()
    missing_data_test = test[numerical_features].isnull().sum()
    print("-------Dữ liệu bị thiếu:")
    print("---Train---")
    print(missing_data_train)

    # Replace
    train[numerical_features] = train[numerical_features].fillna(train[numerical_features].median())
    test[numerical_features] = test[numerical_features].fillna(test[numerical_features].median())

    # Including Zero values
    cols = train[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]

    # Replace
    for col in cols:
        median = train[col].median()
        train[col] = np.where(train[col] <= 0, median, train[col]) # condition true => median else stay train[col]
        test[col] = np.where(test[col] <= 0, median, test[col]) # apply the same to test (transform test from train fitted median

    for feature in numerical_features:
        train[feature] = train[feature].astype(trainCopy[feature].dtype)  # Revert back to original Data type
        test[feature] = test[feature].astype(trainCopy[feature].dtype) # apply the same to test set
    print("-------Điền giá trị thiếu hoàn tất!")

    return train,test