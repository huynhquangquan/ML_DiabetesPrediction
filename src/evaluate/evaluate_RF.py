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
    test = pd.read_csv(base_dir / 'data/processed/test.csv')
    train = pd.read_csv(base_dir / 'data/processed/train.csv')
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
    print("Đánh giá mô hình Random Forest:", scores.mean())

    # Evaluate model on test data
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    print("Độ chính xác trên tập test của mô hình Random Forest:", accuracy)
    print("Precision trên tập test của mô hình Random Forest:", precision)
    print("Recall trên tập test của mô hình Random Forest:", recall)
    print("F1-score trên tập test của mô hình Random Forest:", f1)

    # Create table to save results
    ResultTable = pd.DataFrame(columns= ["Model","Average(CV) Accuracy/E","Test Accuracy","Precision","Recall","f1-score"])

    ResultTable = pd.DataFrame({
        "Model": ["RandomForestClassifier"],
        "Average(CV) Accuracy": [scores.mean()],
        "Test Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1-score": [f1]
    })

    ResultTable.to_csv(base_dir/'results/reports/Kết quả mô hình RF', index=False)
    print("-----Đã lưu kết quả mô hình RF")

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
    loc_dir = dir / '../results/figures/Báo cáo phân loại - Random Forest.png'
    plt.savefig(loc_dir.resolve())
    print("-----Đã lưu figure báo cáo phân loại RF!")