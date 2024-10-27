def remove_outliers(df):
    attributes = df.columns.to_list()
    outlierIndexes = set()
    for attribute in attributes:
        Q1 = df[attribute].describe()["25%"]
        Q3 = df[attribute].describe()["75%"]
        IQR = Q3 - Q1
        Min = Q1 - 1.5 * IQR
        Max = Q3 + 1.5 * IQR
        outlierIndexesOfAtrribute = df[(df[attribute] < Min) | (df[attribute] > Max)].index.to_list()
        outlierIndexes = outlierIndexes.union(set(outlierIndexesOfAtrribute))
    print("-------Loại bỏ giá trị ngoại lai hoàn tất!")
    return df.drop(list(outlierIndexes))