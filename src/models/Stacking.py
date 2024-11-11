import warnings
from src import utilities
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
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
        X_train, bin = utilities.scaling(X_train, bin)
        bin = None

    # Define individual models and load them
    model1, model2, model3 = utilities.model_ensemble()
    model1 = utilities.joblib_load(model1)
    model2 = utilities.joblib_load(model2)
    model3 = utilities.joblib_load(model3)

    # Choose rf as final estimator
    stacking_clf = StackingClassifier(
        estimators=[(f'{model1}', model1), (f'{model2}', model2), (f'{model3}', model3)],
        final_estimator=RandomForestClassifier(),
    )

    stacking_clf.fit(X_train, y_train)

    utilities.joblib_dump(stacking_clf,'Stacking')

    print("-------Lưu mô hình Stacking hoàn tất")