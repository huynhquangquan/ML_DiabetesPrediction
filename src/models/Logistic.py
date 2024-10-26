import warnings
from src import utilities
from pathlib import Path
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # Split train data
    base_dir = Path(__file__).parent.parent

    check_processed = bool(utilities.check_processed())

    if check_processed is False:
        raise RuntimeError("Train mô hình thất bại")

    train = pd.read_csv(base_dir / '..' / 'data' / 'processed' / 'train.csv')

    X_train = train.drop(columns=['Outcome'])
    y_train = train['Outcome']

    X_train = utilities.scaling(X_train)

    model = LogisticRegression(max_iter= 1000,random_state=42)

    # HyperParameters
    param_grid = {
        "solver": ["liblinear"], #"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
        "penalty": ["l1", "l2", "elasticnet", None]
    }

    random_search = GridSearchCV(model,param_grid,cv=5,n_jobs=-1,verbose=2)

    random_search.fit(X_train, y_train)
    print("Best parameters:", random_search.best_params_)

    utilities.joblib_dump(random_search.best_estimator_,'Logistic')

    print("-------Lưu mô hình Logistic hoàn tất")