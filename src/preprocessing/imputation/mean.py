import pandas as pd

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
    zero_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', "Insulin", 'BMI']
    diabetes_mean = train[zero_features].mean()
    train[zero_features] = train[zero_features].replace(0, diabetes_mean)
    test[zero_features] = test[zero_features].replace(0, diabetes_mean)
    for feature in numerical_features:
        train[feature] = train[feature].astype(trainCopy[feature].dtype)
        test[feature] = test[feature].astype(trainCopy[feature].dtype)
    return train,test