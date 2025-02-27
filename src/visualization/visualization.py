import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from src import utilities
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score, classification_report

if __name__=="__main__":
    model_name = utilities.model_select()['name']
    scaling = utilities.scaling_config()
    check_processed = bool(utilities.check_processed())
    check_model = bool(utilities.check_model(model_name))
    if check_processed is False or check_model is False:
        raise RuntimeError("Visualize thất bại")

    # Load processed test data and model
    test = utilities.read_processed("test.csv")
    X_test = test.drop(columns=["Outcome"])
    y_test = test["Outcome"]
    model = utilities.joblib_load(model_name)

    if scaling == "enable":
        X_train = utilities.read_processed('train.csv')
        X_train = X_train.drop(columns=['Outcome'])
        X_train, X_test = utilities.scaling(X_train, X_test)
        X_train = None

    # Create pred variable
    threshold = float(utilities.model_select()['threshold'])
    if threshold > 1 or threshold <= 0:
        raise RuntimeError("Threshold phải từ 0.1 đến 1.0")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Create classification report for Random Forest
    report = classification_report(y_test, y_pred, output_dict=True)

    # Visualize classification report by drawing heatmap chart
    labels = list(report.keys())[:-3]
    data = [[report[label]['precision'], report[label]['recall'], report[label]['f1-score']] for label in labels]
    data_array = np.array(data)
    plt.figure(figsize=(10, 6))
    sns.heatmap(data_array, annot=True, fmt=".2f", xticklabels=['Precision', 'Recall', 'F1-score'], yticklabels=labels, cmap='Blues')
    plt.xlabel('Chỉ số')
    plt.ylabel('Lớp')
    plt.title(f'Báo cáo phân loại - {model_name}')

    dir = Path(__file__).parent.parent
    plt.savefig((dir / '..' / 'results' / 'figures' / f'Báo cáo phân loại - {model_name}.png').resolve())
    print("-----Đã lưu figure báo cáo phân loại")
    plt.close()

    # Visualize confusion matrix of prediction by drawing heatmap
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm,annot=True,fmt='d',cbar=False,cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Dự đoán kết quả - {model_name}')
    plt.savefig((dir / '..' / 'results' / 'figures' / f'Ma trận nhầm lẫn - {model_name}.png').resolve())
    print("-----Đã lưu figure Ma trận nhầm lẫn")
    plt.close()
    # Accuracy: (True Positives + True Negatives) / Total instances
    # Precision for class 1: True Positives / (True Positives + False Positives)
    # Recall for class 1: True Positives / (True Positives + False Negatives)
