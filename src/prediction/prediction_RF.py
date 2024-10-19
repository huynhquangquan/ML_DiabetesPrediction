from src import utilities
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

if __name__=="__main__":
    check_raw = bool(utilities.check_raw())
    check_model = bool(utilities.check_model("random_forest"))
    if check_raw is False and check_model is False:
        print("Prediction RF thất bại")
        sys.exit()

    # Load raw data and model
    model = utilities.joblib_load("random_forest")
    data_samples = utilities.read_raw("diabetes.csv")
    
    # # Create external data for prediction
    # np.random.seed(42)
    # data_samples = pd.DataFrame([
    #     [np.random.randint(0, 12),  # Pregnancies
    #      np.random.randint(70, 200),  # Glucose
    #      np.random.randint(50, 200),  # BloodPressure
    #      np.random.randint(0, 50),  # SkinThickness
    #      np.random.randint(0, 600),  # Insulin
    #      round(np.random.uniform(18, 50), 1),  # BMI
    #      round(np.random.uniform(0.1, 2.5), 3),  # DiabetesPedigreeFunction
    #      np.random.randint(21, 80)]  # Age
    #     for _ in range(10)  # Generate 5 new samples
    # ], columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
    # 
    # # Function for determining diabetes based on conditions given
    # def determine_outcome(row):
    #     if (row['Glucose'] >= 126) and \
    #             (row['DiabetesPedigreeFunction'] >= 0.5) and \
    #             (row['Insulin'] <= 50 or \
    #              row['Insulin'] >= 150) and \
    #             (row['SkinThickness'] >=30) and \
    #              (row['Pregnancies'] >= 5) and \
    #             (row['BMI'] >= 30) and \
    #             (row['Age'] >= 45) and \
    #             (row['BloodPressure'] >= 130):
    #         return 1
    #     else:
    #         return 0
    # Shuffle before defining X and y
    data_samples = data_samples.sample(frac=1)
    # data_samples['Outcome'] = data_samples.apply(determine_outcome, axis=1)

    X = data_samples.drop(columns=["Outcome"])
    y = data_samples["Outcome"]

    # Prediction
    pred = model.predict(X)
    X['Actual'] = y
    X['Predict'] = pred
    print(X)

    # Count Accuracy based on correct predict
    accuracy = (X['Actual']==X['Predict']).sum() / len(X)
    print(f'Accuracy: {accuracy*100:.2f}%')


    dir = Path(__file__).parent.parent
    X.to_csv(dir / '..' / 'results' / 'reports' / 'Pred vs Data.csv')
    print("-----Đã lưu Pred vs Data")