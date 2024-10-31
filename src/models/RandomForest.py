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

    # HyperParameters
    # Define ranges
    # n_estimators_range = np.arange(20, 100)  # 100, 200, 300, 400, 500 #(50, 3001, 50) (1000, 5001, 50)
    # max_depth_range = [None] + list(np.arange(2, 100, 5))  # None, 10, 20, 30, 40, 50    #(2, 101, 5) for better class 0, (2, 101, 4) for better class 1
    # min_samples_split_range = np.arange(2, 201, 3)  # 2, 4, 6, 8, 10  #(2, 301, 3)
    # min_samples_leaf_range = np.arange(2, 201)  # 1, 2, 3, 4
    # max_features_range = ['log2', 0.2, 0.25, 0.5, 0.75]
    # bootstrap_range = [True, False]
    # class_weight = ["balanced", None]
    # criterion = ['entropy', 'gini']

    # param_grid = {
        # 'n_estimators': n_estimators_range.tolist(),
        # 'max_depth': max_depth_range,
        # 'min_samples_split': min_samples_split_range.tolist(),
        # 'min_samples_leaf': min_samples_leaf_range.tolist(),
        # 'max_features': max_features_range,
        # 'bootstrap': bootstrap_range,
        # 'criterion': criterion,
        # 'class_weight': class_weight,
    # }

    # random_search = RandomizedSearchCV(model,param_grid,n_iter=100,cv=8,random_state=42,n_jobs=-1,verbose=1, error_score='raise')

    model.fit(X_train, y_train)

    utilities.joblib_dump(model,'RandomForest')

    print("-------Lưu mô hình RandomForest hoàn tất")