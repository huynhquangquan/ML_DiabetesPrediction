def remove_outliers(train,test):
    numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                          'BMI', 'DiabetesPedigreeFunction', 'Age']
    for feature in numerical_features:
        Q1 = train[feature].quantile(0.25)
        Q3 = train[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        train = train.loc[(train[feature] >= lower_bound) & (train[feature] <= upper_bound)]
        test = test.loc[(test[feature] >= lower_bound) & (test[feature] <= upper_bound)]
    print("-------Loại bỏ giá trị ngoại lai hoàn tất!")
    return train,test