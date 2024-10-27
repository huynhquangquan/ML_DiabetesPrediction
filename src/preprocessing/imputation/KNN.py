from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

def replace_missing_values(DiabetesDataset):
    columnsThatContainMissingValues = ["Glucose", "BloodPressure", "SkinThickness", "Insulin","BMI"]
    DiabetesDatasetCopy = DiabetesDataset.replace({"Glucose" : 0, "BloodPressure" : 0,
                                                   "SkinThickness" : 0, "Insulin" : 0,
                                                   "BMI" : 0}, np.nan)
    imputer = KNNImputer(missing_values=np.nan, n_neighbors = 5)
    ImputedData = imputer.fit_transform(DiabetesDatasetCopy)
    ImputedDataFrame = pd.DataFrame(ImputedData, columns = DiabetesDatasetCopy.columns)
    for column in columnsThatContainMissingValues:
        dtype = DiabetesDatasetCopy[column].dtype
        ImputedDataFrame[column] = ImputedDataFrame[column].astype(dtype)
    return ImputedDataFrame

# def replace_missing_values(df):
#     columnsThatContainMissingValues = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
#
#     # Replace zero values with NaN for imputation
#     df_copy = df.copy()
#     df_copy[columnsThatContainMissingValues] = df_copy[columnsThatContainMissingValues].replace(0, np.nan)
#
#     # Apply KNN Imputer
#     imputer = KNNImputer(n_neighbors=5)
#     df_copy[columnsThatContainMissingValues] = imputer.fit_transform(df_copy[columnsThatContainMissingValues])
#
#     return df_copy