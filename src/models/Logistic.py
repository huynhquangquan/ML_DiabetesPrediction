import warnings
from src import utilities
from pathlib import Path
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np

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
        # Convert X_train_scaled back to a DataFrame
        X_train_scaled, bin = utilities.scaling(X_train, bin)
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    model = LogisticRegression(max_iter= 10000000,random_state=42)

    # HyperParameters
    # Define
    solver = ["lbfgs"]
    penalty = [None]
    C = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    class_weight = [{0:3,1:2}]
    fit_intercept= [True, False, None]
    dual= [True, False, None]
    # max_iter = np.arange(50,100)


    param_grid = {
        "solver": solver,
        "penalty": penalty,
        # "C": C,
        # "class_weight": class_weight,
        # "fit_intercept": fit_intercept,
        # "dual": dual
        # "max_iter": max_iter
    }

    random_search = GridSearchCV(model,param_grid,cv=5,n_jobs=-1,verbose=1)

    random_search.fit(X_train, y_train)
    print("Best parameters:", random_search.best_params_)

    utilities.joblib_dump(random_search.best_estimator_,'Logistic')

    print("-------Lưu mô hình Logistic hoàn tất")