from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from src import utilities
import numpy as np

if __name__ == "__main__":
    model_name = utilities.model_select()
    scaling = utilities.scaling_config()
    check_processed = utilities.check_processed()
    check_model = utilities.check_model(model_name)
    if check_processed is False and check_model is False:
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
        X_train_scaled, X_test_scaled = utilities.scaling(X_train, X_test)
        # Convert X_scaled back to a DataFrame
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    model = utilities.joblib_load(model_name)

    # Create pred variable
    y_pred = model.predict(X_test)

    # Evaluate model with cross_validation
    scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)

    # Evaluate model on test data
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    print(f'Average(CV) Accuracy: {scores.mean():.4f}')
    print(f'Accuracy on Test: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

    # Create table to save results
    ResultTable = pd.DataFrame(columns= ["Model","Average(CV) Accuracy","Accuracy on Test","Precision","Recall","F1-score"])

    ResultTable = pd.DataFrame({
        "Model": [model_name],
        "Average(CV) Accuracy": f'{scores.mean():.2f}',
        "Accuracy on Test": f'{accuracy:.2f}',
        "Precision": f'{precision:.2f}',
        "Recall": f'{recall:.2f}',
        "F1-score": f'{f1:.2f}'
    })

    ResultTable.to_csv(base_dir/f'results/reports/Kết quả mô hình {model_name}.csv', index=False)
    print("-----Đã lưu kết quả mô hình")