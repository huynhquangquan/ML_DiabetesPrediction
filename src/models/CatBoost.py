import warnings
from src import utilities
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from catboost import CatBoostClassifier

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
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
        X_train_scaled, bin = utilities.scaling(X_train, bin)
        bin = None

    model = CatBoostClassifier(random_state=42, silent=True)

    # HyperParameters
    # Define ranges
    # iterations_range = np.arange(100, 1000, 50)  # Number of boosting iterations
    # depth_range = np.arange(4, 10)  # Depth of trees in the model
    # learning_rate_range = np.linspace(0.01, 0.3, 10)  # Learning rate
    # l2_leaf_reg_range = np.arange(1, 10)  # L2 regularization coefficient
    # border_count_range = np.arange(32, 256, 32)  # Number of splits for numerical features
    #
    # param_grid = {
    #     'iterations': iterations_range.tolist(),
    #     'depth': depth_range.tolist(),
    #     'learning_rate': learning_rate_range.tolist(),
    #     'l2_leaf_reg': l2_leaf_reg_range.tolist(),
    #     'border_count': border_count_range.tolist(),
    #     'loss_function': ['Logloss', 'CrossEntropy']
    # }

    # random_search = RandomizedSearchCV(model,param_grid,n_iter=50,cv=5,random_state=42,n_jobs=-1,verbose=1, error_score="raise")

    model.fit(X_train, y_train)
    utilities.joblib_dump(model,'CatBoost')

    print("-------Lưu mô hình CatBoost hoàn tất")