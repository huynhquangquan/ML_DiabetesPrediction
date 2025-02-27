import pandas as pd
from src import utilities
from sklearn.model_selection import train_test_split
from pathlib import Path
from src import features_engineering
import numpy as np

def check_samples(df):
    num_samples = df.shape[0]
    print("-------Số lượng mẫu trong df hiện tại là:", num_samples)

def delete_duplicate_rows(df):
    duplicated_rows = df[df.duplicated()]
    print("-------Các dòng trùng lặp:")
    print(duplicated_rows)
    # Delete duplicate rows
    df_cleaned = df.drop_duplicates()
    print("-------Xóa dữ liệu lặp hoàn tất!")
    return df_cleaned

def split_data(df):
    # Split to train and test data
    train,test = train_test_split(df, test_size=0.2, random_state=42)
    print("-------Chia dữ liệu thành train data và test data thành công")
    return train,test

def save_data_preprocessed(train,test):
    Base_dir = Path(__file__).parent.parent
    # NOTE: A must whenever saving processed data, index always False, else it will include index as a column in data
    train.to_csv(Base_dir / '..' / 'data' / 'processed' / 'train.csv',index=False)
    test.to_csv(Base_dir / '..' / 'data' / 'processed' / 'test.csv', index=False)
    print("-------Lưu data huấn luyện sau khi xử lý")
    print("-------Lưu hoàn tất!")

if __name__ == "__main__":
    imputation_select = utilities.imputation_select()
    balance_select = utilities.balance_select()
    outliers_select = utilities.outliers_select()
    dataset = utilities.dataset_select()['dataset']
    check_raw = bool(utilities.check_raw(dataset))

    if check_raw is False:
        raise RuntimeError("Preprocessing thất bại")

    # Preprocess dataset
    print("======================Dataset==============================================================================================================")
    df = utilities.read_raw(dataset)
    print("Dữ liệu dataset:")
    print(df)
    check_samples(df)
    removing_features = features_engineering.check_correlation(df,0.8, "Outcome")
    df = features_engineering.remove_features(df, removing_features)

    print("----------------------Xóa dữ liệu lặp--------------------------------------------------------------------------------------------------------------")
    df = delete_duplicate_rows(df)
    print("Dữ liệu dataset sau khi xóa dữ liệu lặp:")
    print(df)
    check_samples(df)

    print("----------------------Chia dữ liệu--------------------------------------------------------------------------------------------------------------")
    train,test = split_data(df) # Get train data, save test data
    print("Dữ liệu huấn luyện:")
    print(train)
    check_samples(train)
    print("Dữ liệu test:")
    print(test)
    check_samples(test)

    print("----------------------Điền giá trị thiếu--------------------------------------------------------------------------------------------------------------")
    try:
        impute_function = utilities.dynamic_import_imputation(imputation_select)  # Dynamically get the imputation function
        imputation_df, bin = impute_function(df, df.copy())
        bin = None
    except Exception as e:
        print(f"Chưa chọn imputation: {e}")
    print("Dữ liệu huấn luyện sau khi điền:")
    print(imputation_df)
    check_samples(imputation_df)

    if outliers_select == "None":
        dfpr_dir = Path(__file__).parent.parent
        imputation_df.to_csv(dfpr_dir / '..' / 'data' / 'processed' / dataset, index=False)
    else:
        print("----------------------Xử lý giá trị ngoại lai--------------------------------------------------------------------------------------------------------------")
        outliers_function = utilities.dynamic_import_outliers(outliers_select)  # Dynamically get the imputation function
        rmoutliers_df, bin = outliers_function(imputation_df, imputation_df.copy())
        bin = None
        print("Dữ liệu huấn luyện sau khi loại bỏ ngoại lai:")
        print(rmoutliers_df)
        check_samples(rmoutliers_df)
        dfpr_dir = Path(__file__).parent.parent
        rmoutliers_df.to_csv(dfpr_dir / '..' / 'data' / 'processed' / dataset, index=False)

    # Preprocess Train set
    print("======================Train==============================================================================================================")
    print("----------------------Điền giá trị thiếu--------------------------------------------------------------------------------------------------------------")
    try:
        impute_function = utilities.dynamic_import_imputation(imputation_select)  # Dynamically get the imputation function
        imputation_train, imputation_test = impute_function(train, test)
    except Exception as e:
        print(f"Chưa chọn imputation: {e}")
    print("Dữ liệu huấn luyện sau khi điền:")
    print(imputation_train)
    check_samples(imputation_train)
    final_train = imputation_train
    final_test = imputation_test

    if outliers_select == "None":
        pass
    else:
        print("----------------------Xử lý giá trị ngoại lai--------------------------------------------------------------------------------------------------------------")
        outliers_function = utilities.dynamic_import_outliers(outliers_select)  # Dynamically get the outliers function
        rmoutliers_train, rmoutliers_test = outliers_function(final_train, final_test)
        print("Dữ liệu huấn luyện sau khi loại bỏ ngoại lai:")
        print(rmoutliers_train)
        check_samples(rmoutliers_train)
        final_train = rmoutliers_train
        final_test = rmoutliers_test

    if balance_select == "None":
        pass
    else:
        print("----------------------Cân bằng huấn luyện--------------------------------------------------------------------------------------------------------------")
        balance_function = utilities.dynamic_import_balance(balance_select)  # Dynamically get the balance function
        balanced_train = balance_function(final_train,"Outcome")
        print("Dữ liệu huấn luyện sau SMOTE:")
        print(balanced_train)
        check_samples(balanced_train)
        final_train = balanced_train

    # Preprocess Test set
    print("======================TEST==============================================================================================================")
    print("----------------------Điền giá trị thiếu--------------------------------------------------------------------------------------------------------------")
    print("Dữ liệu huấn luyện sau khi điền:") # Test has already been transformed when fit imputating Train data.
    print(imputation_test)
    check_samples(imputation_test)

    if outliers_select == "None":
        pass
    else:
        print("----------------------Xử lý giá trị ngoại lai--------------------------------------------------------------------------------------------------------------")
        print("Dữ liệu huấn luyện sau khi loại bỏ ngoại lai:")
        print(rmoutliers_test)
        check_samples(rmoutliers_test)

    # Save Data
    print("====================================================================================================================================")
    print("-------Dataset sau khi xử lý: ")
    print("TRAIN SET:")
    print(final_train)
    print("TEST SET")
    print(final_test)
    save_data_preprocessed(final_train,final_test)





