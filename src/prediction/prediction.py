from src import utilities
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

if __name__=="__main__":
    model_name = utilities.model_select()['name']
    dataset = utilities.dataset_select()['dataset']
    scaling = utilities.scaling_config()
    check_raw = bool(utilities.check_raw(dataset))
    check_processed = bool(utilities.check_processed())
    check_external = bool(utilities.check_external())
    check_model = bool(utilities.check_model(model_name))
    if check_raw is False or check_model is False or check_processed is False or check_external is False:
        raise RuntimeError("Prediction thất bại")

    # Load data-samples data and model
    model = utilities.joblib_load(model_name)
    data_samples_processed = utilities.read_processed(dataset)
    data_samples_raw = utilities.read_raw(dataset)
    data_sample_external = utilities.read_processed('test.csv')
    # Shuffle before defining X and y
    data_samples_processed = data_samples_processed.sample(frac=1)
    data_samples_external = data_sample_external.sample(frac=1)
    data_samples_raw = data_samples_raw.sample(frac=1)

    X_processed = data_samples_processed.drop(columns=["Outcome"])
    y_processed = data_samples_processed["Outcome"]
    X_raw = data_samples_raw.drop(columns=["Outcome"])
    y_raw = data_samples_raw["Outcome"]
    X_external = data_samples_external.drop(columns=["Outcome"])
    y_external = data_samples_external["Outcome"]

    threshold = float(utilities.model_select()['threshold'])
    if threshold > 1 or threshold <= 0:
        raise RuntimeError("Threshold phải từ 0.1 đến 1.0")

    if scaling == "enable":
        bin = X_processed.copy()
        X_processed, bin = utilities.scaling(X_processed,bin)
        X_raw, bin = utilities.scaling(X_raw,bin)
        X_external, bin = utilities.scaling(X_external,bin)
        bin = None
    else:
        X_processed = X_processed
        X_raw = X_raw
        X_external = X_external

    y_proba = model.predict_proba(X_processed)[:, 1]
    pred_processed = (y_proba >= threshold).astype(int)
    y_proba = model.predict_proba(X_raw)[:, 1]
    pred_raw = (y_proba >= threshold).astype(int)
    y_proba = model.predict_proba(X_external)[:, 1]
    pred_external = (y_proba >= threshold).astype(int)

    X_processed['Actual'] = y_processed
    X_processed['Predict'] = pred_processed
    X_raw['Actual'] = y_raw
    X_raw['Predict'] = pred_raw
    X_external['Actual'] = y_external
    X_external['Predict'] = pred_external

    # Print and export results of prediction
    # PROCESSED
    print("Pred vs Processed Dataset")
    print(X_processed)
    accuracy_processed = (X_processed['Actual']==X_processed['Predict']).sum() / len(X_processed)
    accuracy_on_neg_ppr = ((X_processed['Actual']==0) & (X_processed['Predict']==0)).sum()  / (X_processed['Actual']==0).sum()
    accuracy_on_pos_ppr = ((X_processed['Actual']==1) & (X_processed['Predict']==1)).sum() / (X_processed['Actual']==1).sum()
    print(f'Accuracy: {accuracy_processed*100:.2f}%')
    print(f'Accuracy on Not Diabetes: {accuracy_on_neg_ppr*100:.2f}%')
    print(f'Accuracy on Diabetes: {accuracy_on_pos_ppr*100:.2f}%')

    # RAW
    print("Pred vs Raw Dataset")
    print(X_raw)
    accuracy_raw = (X_raw['Actual']==X_raw['Predict']).sum() / len(X_raw)
    accuracy_on_neg_r = ((X_raw['Actual']==0) & (X_raw['Predict']==0)).sum()  / (X_raw['Actual']==0).sum()
    accuracy_on_pos_r = ((X_raw['Actual']==1) & (X_raw['Predict']==1)).sum() / (X_raw['Actual']==1).sum()
    print(f'Accuracy: {accuracy_raw*100:.2f}%')
    print(f'Accuracy on Not Diabetes: {accuracy_on_neg_r*100:.2f}%')
    print(f'Accuracy on Diabetes: {accuracy_on_pos_r*100:.2f}%')

    # EXTERNAL
    print("Pred vs External Dataset")
    print(X_external)
    accuracy_external = (X_external['Actual']==X_external['Predict']).sum() / len(X_external)
    accuracy_on_neg_xtn = ((X_external['Actual']==0) & (X_external['Predict']==0)).sum()  / (X_external['Actual']==0).sum()
    accuracy_on_pos_xtn = ((X_external['Actual']==1) & (X_external['Predict']==1)).sum() / (X_external['Actual']==1).sum()
    print(f'Accuracy: {accuracy_external*100:.2f}%')
    print(f'Accuracy on Not Diabetes: {accuracy_on_neg_xtn*100:.2f}%')
    print(f'Accuracy on Diabetes: {accuracy_on_pos_xtn*100:.2f}%')

    results = pd.DataFrame({
        f'{model_name}': ["Accuracy on Dataset", "Accuracy on Not Diabetes", "Accuracy on Diabetes"],
        "Processed": [
            f'{accuracy_processed * 100:.2f}%',
            f'{accuracy_on_neg_ppr * 100:.2f}%',
            f'{accuracy_on_pos_ppr * 100:.2f}%'
        ],
        "Raw": [
            f'{accuracy_raw * 100:.2f}%',
            f'{accuracy_on_neg_r * 100:.2f}%',
            f'{accuracy_on_pos_r * 100:.2f}%'
        ],
        "External": [
            f'{accuracy_external * 100:.2f}%',
            f'{accuracy_on_neg_xtn * 100:.2f}%',
            f'{accuracy_on_pos_xtn * 100:.2f}%'
        ]
    })

    dir = Path(__file__).parent.parent
    results.to_csv(dir / '..' / 'results' / 'reports' / f'PredvsData_{model_name}.csv',index=False)
    print("-----Đã lưu PredvsData")