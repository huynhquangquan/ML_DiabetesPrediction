import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from src import utilities
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score, classification_report

if __name__=="__main__":
    check_processed = bool(utilities.check_processed())
    check_model = bool(utilities.check_model("random_forest"))
    if check_processed is False and check_model is False:
        raise RuntimeError("Visualize RF thất bại")

    # Load processed data and model
    # train = utilities.read_processed("train.csv")
    test = utilities.read_processed("test.csv")
    # X_train = train.drop(columns=["Outcome"])
    # y_train = train["Outcome"]
    X_test = test.drop(columns=["Outcome"])
    y_test = test["Outcome"]
    model = utilities.joblib_load("random_forest")

    y_pred = model.predict(X_test)
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
    plt.title('Báo cáo phân loại - Random Forest')

    dir = Path(__file__).parent.parent
    plt.savefig((dir / '..' / 'results' / 'figures' / 'Báo cáo phân loại - Random Forest.png').resolve())
    print("-----Đã lưu figure báo cáo phân loại RF")
    plt.close()

    # Visualize confusion matrix of prediction by drawing heatmap
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm,annot=True,fmt='d',cbar=False,cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Dự đoán kết quả - Random Forest')
    plt.savefig((dir / '..' / 'results' / 'figures' / 'Ma trận nhầm lẫn - Random Forest.png').resolve())
    print("-----Đã lưu figure Ma trận nhầm lẫn RF")
    plt.close()
    # Accuracy: (True Positives + True Negatives) / Total instances
    # Precision for class 1: True Positives / (True Positives + False Positives)
    # Recall for class 1: True Positives / (True Positives + False Negatives)
