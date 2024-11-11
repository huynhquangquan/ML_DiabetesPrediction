from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from src import utilities
import numpy as np

if __name__ == "__main__":
    model_name = utilities.model_select()['name']
    scaling = utilities.scaling_config()
    check_processed = utilities.check_processed()
    check_model = utilities.check_model(model_name)
    if check_processed is False or check_model is False:
        raise RuntimeError("Evaluate thất bại")

    # Load data and model
    base_dir = Path(__file__).parent.parent.parent
    test = pd.read_csv(base_dir / 'data' / 'processed' / 'test.csv')
    train = pd.read_csv(base_dir / 'data' / 'processed' / 'train.csv')
    X_train = train.drop(columns=['Outcome'])
    y_train = train['Outcome']
    X_test = test.drop(columns=['Outcome'])
    y_test = test['Outcome']
    if scaling == "enable":
        X_train, X_test = utilities.scaling(X_train, X_test)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    model = utilities.joblib_load(model_name)

    # Create pred variable
    threshold = float(utilities.model_select()['threshold'])
    if threshold > 1 or threshold <= 0:
        raise RuntimeError("Threshold phải từ 0.1 đến 1.0")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Evaluate model with cross_validation
    scores_auc = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1, scoring= "roc_auc")
    scores_accuracy = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1, scoring= "accuracy")

    # Evaluate model on test data
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    print(f'Average(CV) ROC_AUC: {scores_auc.mean():.4f}')
    print(f'Average(CV) Accuracy: {scores_accuracy.mean():.4f}')
    print(f'Accuracy on Test: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

    # Create table to save results
    ResultTable = pd.DataFrame({
        "Model": [model_name],
        "Average(CV) ROC_AUC": f'{scores_auc.mean():.2f}',
        "Average(CV) Accuracy": f'{scores_accuracy.mean():.2f}',
        "Accuracy on Test": f'{accuracy:.2f}',
        "Precision": f'{precision:.2f}',
        "Recall": f'{recall:.2f}',
        "F1-score": f'{f1:.2f}',
        "Imputation": f'{utilities.imputation_select()}',
        "Outliers": f'{utilities.outliers_select()}',
        "Balance": f'{utilities.balance_select()}',
        "Scaling": f'{utilities.scaler_select()}' if scaling == "enable" else 'None'
    })

    ResultTable.to_csv(base_dir/f'results/reports/Kết quả mô hình {model_name}.csv', index=False)
    print("-----Đã lưu kết quả mô hình")