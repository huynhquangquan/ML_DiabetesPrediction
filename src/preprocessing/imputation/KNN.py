from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np


def replace_missing_values(train, test):
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
            'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

    # Replace 0s with NaN in specific columns
    trainCopy = train.replace({"Glucose": 0, "BloodPressure": 0, "SkinThickness": 0,
                               "Insulin": 0, "BMI": 0}, np.nan)
    testCopy = test.replace({"Glucose": 0, "BloodPressure": 0, "SkinThickness": 0,
                             "Insulin": 0, "BMI": 0}, np.nan)

    imputer = KNNImputer(missing_values=np.nan, n_neighbors=5)
    missing_values = trainCopy[cols].isnull().sum()
    print("-------Dữ liệu bị thiếu:")
    print(missing_values)

    # Separate diabetic and non-diabetic data
    train_diabetic = trainCopy[trainCopy['Outcome'] == 1]
    train_non_diabetic = trainCopy[trainCopy['Outcome'] == 0]
    test_diabetic = testCopy[testCopy['Outcome'] == 1]
    test_non_diabetic = testCopy[testCopy['Outcome'] == 0]

    # Apply KNNImputer on the separate datasets
    ImputedTrainData_1 = imputer.fit_transform(train_diabetic.drop('Outcome', axis=1))
    ImputedTestData_1 = imputer.transform(test_diabetic.drop('Outcome', axis=1))
    ImputedTrainData_0 = imputer.fit_transform(train_non_diabetic.drop('Outcome', axis=1))
    ImputedTestData_0 = imputer.transform(test_non_diabetic.drop('Outcome', axis=1))

    # Reconstruct the DataFrames with the imputed data
    ImputedTrainDataFrame_1 = pd.DataFrame(ImputedTrainData_1, columns=trainCopy.drop('Outcome', axis=1).columns)
    ImputedTestDataFrame_1 = pd.DataFrame(ImputedTestData_1, columns=testCopy.drop('Outcome', axis=1).columns)
    ImputedTrainDataFrame_0 = pd.DataFrame(ImputedTrainData_0, columns=trainCopy.drop('Outcome', axis=1).columns)
    ImputedTestDataFrame_0 = pd.DataFrame(ImputedTestData_0, columns=testCopy.drop('Outcome', axis=1).columns)

    # Add the 'Outcome' column back
    ImputedTrainDataFrame_1['Outcome'] = train_diabetic['Outcome'].values
    ImputedTestDataFrame_1['Outcome'] = test_diabetic['Outcome'].values
    ImputedTrainDataFrame_0['Outcome'] = train_non_diabetic['Outcome'].values
    ImputedTestDataFrame_0['Outcome'] = test_non_diabetic['Outcome'].values

    # Concatenate the diabetic and non-diabetic data
    ImputedTrainDataFrame = pd.concat([ImputedTrainDataFrame_1, ImputedTrainDataFrame_0])
    ImputedTestDataFrame = pd.concat([ImputedTestDataFrame_1, ImputedTestDataFrame_0])

    # Ensure the correct data types
    for column in cols:
        dtype = train[column].dtype
        ImputedTrainDataFrame[column] = ImputedTrainDataFrame[column].astype(dtype)
        ImputedTestDataFrame[column] = ImputedTestDataFrame[column].astype(dtype)

    return ImputedTrainDataFrame, ImputedTestDataFrame
