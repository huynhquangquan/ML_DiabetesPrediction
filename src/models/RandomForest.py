import warnings
from src import utilities
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

if __name__ == "__main__":
    # warnings.filterwarnings('ignore')
    scaling = utilities.scaling_config()
    # Split train data
    base_dir = Path(__file__).parent.parent

    check_processed = bool(utilities.check_processed())

    if check_processed is False:
        raise RuntimeError("Train mô hình thất bại")

    train = pd.read_csv(base_dir / '..' / 'data' / 'processed' / 'train.csv')

    X_train = train.drop(columns=['Outcome'])
    y_train = train['Outcome']

    if scaling == "enable":
        bin = X_train.copy()
        X_train, bin = utilities.scaling(X_train, bin)
        bin = None

    model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)
    # print(model.best_params_)

    utilities.joblib_dump(model,'RandomForest')

    print("-------Lưu mô hình RandomForest hoàn tất")