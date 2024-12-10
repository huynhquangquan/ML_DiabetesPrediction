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
    check_model = bool(utilities.check_model(model_name))
    if check_raw is False or check_model is False or check_processed is False:
        raise RuntimeError("Prediction thất bại")

    # Load data-samples data and model
    model = utilities.joblib_load(model_name)
    data_samples_processed = utilities.read_processed(dataset)
    data_samples_raw = utilities.read_raw(dataset)
    data_test = utilities.read_processed('test.csv')
    # data_test = utilities.read_test()
    # Shuffle before defining X and y
    data_samples_processed = data_samples_processed.sample(frac=1)
    data_test = data_test.sample(frac=1)
    data_samples_raw = data_samples_raw.sample(frac=1)

    X_processed = data_samples_processed.drop(columns=["Outcome"])
    y_processed = data_samples_processed["Outcome"]
    X_raw = data_samples_raw.drop(columns=["Outcome"])
    y_raw = data_samples_raw["Outcome"]
    X_test = data_test.drop(columns=["Outcome"])
    y_test = data_test["Outcome"]

    threshold = float(utilities.model_select()['threshold'])
    if threshold > 1 or threshold <= 0:
        raise RuntimeError("Threshold phải từ 0.1 đến 1.0")

    if scaling == "enable":
        X_train = utilities.read_processed('train.csv')
        X_train = X_train.drop(columns=['Outcome'])
        bin, X_processed_scaled = utilities.scaling(X_train,X_processed)
        bin, X_raw_scaled = utilities.scaling(X_train,X_raw)
        bin, X_test_scaled = utilities.scaling(X_train,X_test)
        X_train, bin = None, None
    else:
        X_processed_scaled = X_processed
        X_raw_scaled = X_raw
        X_test_scaled = X_test

    y_proba = model.predict_proba(X_processed_scaled)[:, 1]
    pred_processed = (y_proba >= threshold).astype(int)
    y_proba = model.predict_proba(X_raw_scaled)[:, 1]
    pred_raw = (y_proba >= threshold).astype(int)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    pred_test = (y_proba >= threshold).astype(int)

    X_processed['Actual'] = y_processed
    X_processed['Predict'] = pred_processed
    X_raw['Actual'] = y_raw
    X_raw['Predict'] = pred_raw
    X_test['Actual'] = y_test
    X_test['Predict'] = pred_test

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

    # TEST
    print("Preview: Pred vs Test Dataset")
    print(X_test)
    accuracy_test = (X_test['Actual']==X_test['Predict']).sum() / len(X_test)
    accuracy_on_neg_xtn = ((X_test['Actual']==0) & (X_test['Predict']==0)).sum()  / (X_test['Actual']==0).sum()
    accuracy_on_pos_xtn = ((X_test['Actual']==1) & (X_test['Predict']==1)).sum() / (X_test['Actual']==1).sum()
    print(f'Accuracy: {accuracy_test*100:.2f}%')
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
        "Preview Test": [
            f'{accuracy_test * 100:.2f}%',
            f'{accuracy_on_neg_xtn * 100:.2f}%',
            f'{accuracy_on_pos_xtn * 100:.2f}%'
        ]
    })

    dir = Path(__file__).parent.parent
    results.to_csv(dir / '..' / 'results' / 'reports' / f'PredvsData_{model_name}.csv',index=False, sep='|')
    print("-----Đã lưu PredvsData")