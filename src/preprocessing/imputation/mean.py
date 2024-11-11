import pandas as pd
import numpy as np

def replace_missing_values(train,test):
    trainCopy = train.copy()
    numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                          'BMI', 'DiabetesPedigreeFunction', 'Age']
    # Check
    missing_data_train = train[numerical_features].isnull().sum()
    missing_data_test = train[numerical_features].isnull().sum()
    print("-------Dữ liệu bị thiếu:")
    print(missing_data_train)

    # Replace
    train[numerical_features] = train[numerical_features].fillna(train[numerical_features].mean())
    test[numerical_features] = test[numerical_features].fillna(test[numerical_features].mean())

    # Include zero and negative
    zero_features = train[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', "Insulin", 'BMI']]
    for feature in zero_features:
        mean = train[feature].mean()
        train[feature] = np.where(train[feature] <= 0, mean, train[feature])
        test[feature] = np.where(test[feature] <= 0, mean, test[feature])
    for feature in numerical_features:
        train[feature] = train[feature].astype(trainCopy[feature].dtype)
        test[feature] = test[feature].astype(trainCopy[feature].dtype)
    train['BMI'] = train['BMI'].round(1)
    test['BMI'] = test['BMI'].round(1)
    print("-------Điền giá trị thiếu hoàn tất!")
    return train,test