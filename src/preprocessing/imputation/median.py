import pandas as pd
import numpy as np

def replace_missing_values(train, test):
    trainCopy = train.copy()
    numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    # Replace zero and negative values with NaN to prevent them from affecting median calculation
    features_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age"]
    for feature in features_to_replace:
        train[feature] = train[feature].apply(lambda x: np.nan if x <= 0 else x)
        test[feature] = test[feature].apply(lambda x: np.nan if x <= 0 else x)

    missing_values = train[numerical_features].isnull().sum()
    print("-------Dữ liệu bị thiếu:")
    print(missing_values)

    # Separate the data based on Outcome for different median calculations
    train_diabetic = train[train['Outcome'] == 1]
    train_non_diabetic = train[train['Outcome'] == 0]

    # Calculate median values separately for diabetic and non-diabetic groups
    median_values_diabetic = train_diabetic[numerical_features].median()
    median_values_non_diabetic = train_non_diabetic[numerical_features].median()

    # Impute missing values for diabetic patients using the diabetic median
    train.loc[train['Outcome'] == 1, numerical_features] = \
        train.loc[train['Outcome'] == 1, numerical_features].fillna(median_values_diabetic)
    test.loc[test['Outcome'] == 1, numerical_features] = \
        test.loc[test['Outcome'] == 1, numerical_features].fillna(median_values_diabetic)

    # Impute missing values for non-diabetic patients using the non-diabetic median
    train.loc[train['Outcome'] == 0, numerical_features] = \
        train.loc[train['Outcome'] == 0, numerical_features].fillna(median_values_non_diabetic)
    test.loc[test['Outcome'] == 0, numerical_features] = \
        test.loc[test['Outcome'] == 0, numerical_features].fillna(median_values_non_diabetic)

    # Ensure consistent data types with original
    for feature in numerical_features:
        train[feature] = train[feature].astype(trainCopy[feature].dtype)
        test[feature] = test[feature].astype(trainCopy[feature].dtype)

    print("-------Điền giá trị thiếu hoàn tất!")
    return train, test


# SUMMARY OF SLASH (\)
#The line is long and splitting it into two lines makes it more readable.
# Without the backslash, Python would think the first line ends at the assignment (=),
# causing a syntax error because it expects something to follow.

#The backslash is telling Python that the assignment operation (=) and the following method call (fillna(...))
#should be treated as part of the same logical line of code,
#even though they span multiple lines.

#in short,
#test.loc[test['Outcome'] == 0, numerical_features] = test.loc[test['Outcome'] == 0, numerical_features].fillna(mean_values_non_diabetic)
# is as same as: test.loc[test['Outcome'] == 0, numerical_features] = \
#         test.loc[test['Outcome'] == 0, numerical_features].fillna(mean_values_non_diabetic)
# That's how the \ works in python as a command connecting