# Check for duplicate data, if found, delete them.
import pandas as pd
from src import utilities
from sklearn.model_selection import train_test_split
from pathlib import Path
from src import features_engineering

def check_samples(df):
    num_samples = df.shape[0]
    print("-------Số lượng mẫu trong df hiện tại là:", num_samples)

def delete_duplicate_rows():
    duplicated_rows = df[df.duplicated()]
    print("-------Các dòng trùng lặp:")
    print(duplicated_rows)
    # Delete duplicate rows
    df_cleaned = df.drop_duplicates()
    print("-------Xóa dữ liệu lặp hoàn tất!")


def replace_missing_values():
    numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                          'BMI', 'DiabetesPedigreeFunction', 'Age']
    # Check
    missing_data = df[numerical_features].isnull().sum()
    print("-------Dữ liệu bị thiếu:")
    print(missing_data)
    # Replace
    df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())

    # Including Zero values
    # Check
    cols = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]
    for col in cols:
        zero_values = len(df[df[col] <= 0])
        print("-------Các cột có dữ liệu bằng 0 hoặc dưới 0 của thuộc tính {} là {}".format(col, zero_values))

    # Replace
    cols = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]
    for col in cols:
        df[col] = df[col].astype(float) # Convert to float to avoid FutureWarning, eventhough it does not affect the progress
        median = df[col].median()
        df.loc[df[col] <= 0, col] = median
    print("-------Điền giá trị thiếu hoàn tất!")

def remove_outliers(df):
    numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                          'BMI', 'DiabetesPedigreeFunction', 'Age']
    df_clean = df.copy()
    for feature in numerical_features:
        Q1 = df_clean[feature].quantile(0.25)
        Q3 = df_clean[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[feature] >= lower_bound) & (df_clean[feature] <= upper_bound)]
    print("-------Loại bỏ giá trị ngoại lai hoàn tất!")
    return df_clean

def save_data_preprocessed(df):
    # Split to train and test data
    train,test = train_test_split(df, test_size=0.2, random_state=42)
    print("-------Chia dữ liệu thành train data và test data thành công")

    Base_dir = Path(__file__).parent.parent
    # NOTE: A must whenever saving processed data, index always False, else it will include index as a column in data
    train.to_csv(Base_dir / '..' / 'data' / 'processed' / 'train.csv',index=False)
    test.to_csv(Base_dir / '..' / 'data' / 'processed' / 'test.csv',index=False)
    print("-------Lưu data sau khi xử lý")
    print("-------Lưu hoàn tất!")

if __name__ == "__main__":
    check_raw = bool(utilities.check_raw())
    if check_raw is False:
        sys.exit()
    df = utilities.read_raw('diabetes.csv')
    print(df)
    check_samples(df)
    delete_duplicate_rows()
    replace_missing_values()
    df_processed = remove_outliers(df)
    check_samples(df_processed)
    removing_features = features_engineering.check_correlation(df_processed,0.8)
    df_preprocessed = features_engineering.remove_features(df_processed, removing_features)
    print("-------Dataset sau khi xử lý: ")
    print(df_preprocessed)
    save_data_preprocessed(df_preprocessed)





