from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from src import utilities
import numpy as np

if __name__ == "__main__":
    # Load data and model
    base_dir = Path(__file__).parent.parent.parent
    test = pd.read_csv(base_dir / 'data' / 'processed' / 'test.csv')
    train = pd.read_csv(base_dir / 'data' / 'processed' / 'train.csv')
    X_train = train.drop(columns=['Outcome'])
    y_train = train['Outcome']
    X_test = test.drop(columns=['Outcome'])
    y_test = test['Outcome']
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    model = utilities.joblib_load("random_forest")

    # prediction on test data
    y_pred = model.predict(X_test)

    # Evaluate model with cross_validation
    scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)

    # Evaluate model on test data
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)

    # Create table to save results
    ResultTable = pd.DataFrame(columns= ["Model","Average(CV) Accuracy","Test Accuracy","Precision","Recall","f1-score"])

    ResultTable = pd.DataFrame({
        "Model": ["RandomForestClassifier"],
        "Average(CV) Accuracy": f'{scores.mean():.2f}',
        "Test Accuracy": f'{accuracy:.2f}',
        "Precision": f'{precision:.2f}',
        "Recall": f'{recall:.2f}',
        "F1-score": f'{f1:.2f}'
    })

    ResultTable.to_csv(base_dir/'results/reports/Kết quả mô hình RF.csv', index=False)
    print("-----Đã lưu kết quả mô hình RF")