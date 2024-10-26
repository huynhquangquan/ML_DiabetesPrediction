from src import utilities
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

if __name__=="__main__":
    model_name = utilities.model_select()
    dataset = utilities.dataset_select()['dataset']

    check_raw = bool(utilities.check_raw())
    check_model = bool(utilities.check_model(model_name))
    if check_raw is False and check_model is False:
        raise RuntimeError("Prediction RF thất bại")

    # Load data-samples data and model
    model = utilities.joblib_load(model_name)
    data_samples = utilities.read_processed(dataset)
    # Shuffle before defining X and y
    data_samples = data_samples.sample(frac=1)

    X = data_samples.drop(columns=["Outcome"])
    y = data_samples["Outcome"]

    # Prediction
    pred = model.predict(X)
    X['Actual'] = y
    X['Predict'] = pred

    print("Pred vs External")
    print(X)

    # Count external Accuracy based on correct predict
    accuracy = (X['Actual']==X['Predict']).sum() / len(X)
    accuracy_on_neg = ((X['Actual']==0) & (X['Predict']==0)).sum()  / (X['Actual']==0).sum()
    accuracy_on_pos = ((X['Actual']==1) & (X['Predict']==1)).sum() / (X['Actual']==1).sum()
    print(f'Accuracy: {accuracy*100:.2f}%')
    print(f'Accuracy on Not Diabetes: {accuracy_on_neg*100:.2f}%')
    print(f'Accuracy on Diabetes: {accuracy_on_pos*100:.2f}%')

    results = pd.DataFrame({
        "Model": ["RandomForestClassifier"],
        "Accuracy on External": [f'{accuracy*100:.2f}%'],
        "Accuracy on Not Diabetes": [f'{accuracy_on_neg*100:.2f}%'],
        "Accuracy on Diabetes": [f'{accuracy_on_pos*100:.2f}%']
    })

    dir = Path(__file__).parent.parent
    results.to_csv(dir / '..' / 'results' / 'reports' / 'PredvsData_RF.csv',index=False)
    print("-----Đã lưu PredvsData_RF")