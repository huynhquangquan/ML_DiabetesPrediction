def remove_outliers(train,test):
    attributes = train.columns.to_list()
    outlierIndexes_train = set()
    outlierIndexes_test = set()
    for attribute in attributes:
        Q1 = train[attribute].quantile(0.25)
        Q3 = train[attribute].quantile(0.75)
        IQR = Q3 - Q1
        Min = Q1 - 1.5 * IQR
        Max = Q3 + 1.5 * IQR
        outlierIndexesOfAtrribute_train = train[(train[attribute] < Min) | (train[attribute] > Max)].index.to_list()
        outlierIndexesOfAtrribute_test = test[(test[attribute] < Min) | (test[attribute] > Max)].index.to_list()
        outlierIndexes_train = outlierIndexes_train.union(set(outlierIndexesOfAtrribute_train))
        outlierIndexes_test = outlierIndexes_test.union(set(outlierIndexesOfAtrribute_test))
    print("-------Loại bỏ giá trị ngoại lai hoàn tất!")
    return train.drop(list(outlierIndexes_train)), test.drop(list(outlierIndexes_test))