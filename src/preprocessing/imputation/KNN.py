from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

def replace_missing_values(train,test):
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                          'BMI', 'DiabetesPedigreeFunction', 'Age']
    trainCopy = train.replace({"Glucose" : 0, "BloodPressure" : 0,
                                                   "SkinThickness" : 0, "Insulin" : 0,
                                                   "BMI" : 0}, np.nan)
    testCopy = test.replace({"Glucose": 0, "BloodPressure": 0,
                               "SkinThickness": 0, "Insulin": 0,
                               "BMI": 0}, np.nan)
    imputer = KNNImputer(missing_values=np.nan, n_neighbors = 5)
    ImputedTrainData = imputer.fit_transform(trainCopy)
    ImputedTestData = imputer.transform(testCopy)

    ImputedTrainDataFrame = pd.DataFrame(ImputedTrainData, columns = trainCopy.columns)
    ImputedTestDataFrame = pd.DataFrame(ImputedTestData, columns = testCopy.columns)
    for column in cols:
        dtype = train[column].dtype
        ImputedTrainDataFrame[column] = ImputedTrainDataFrame[column].astype(dtype)
        ImputedTestDataFrame[column] = ImputedTestDataFrame[column].astype(dtype)
    ImputedTrainDataFrame['BMI'] = ImputedTrainDataFrame['BMI'].round(1)
    ImputedTestDataFrame['BMI'] = ImputedTestDataFrame['BMI'].round(1)
    return ImputedTrainDataFrame, ImputedTestDataFrame