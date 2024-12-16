import uvicorn
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware  # Post request for HTML
from src import utilities

def API():
    class Diabetes(BaseModel):
        Pregnancies: int
        Glucose: int
        BloodPressure: int
        SkinThickness: int
        Insulin: float
        BMI: float
        DiabetesPedigreeFunction: float
        Age: int

    app = FastAPI()

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # You can specify allowed origins instead of '*' for better security
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
        allow_headers=["*"],  # Allows all headers
    )  # Connect HTML

    # Load YAML config
    name = utilities.model_select()['name']
    threshold = float(utilities.model_select()['threshold'])

    # Check whether the model exists or not
    check = bool(utilities.check_model(name))
    if check is False:
        return None

    full_pipeline = utilities.joblib_load(f'{name}')

    # Define feature limits
    feature_limits = {
        "Pregnancies": (0, 20),
        "Glucose": (50, 300),
        "BloodPressure": (30, 200),
        "SkinThickness": (0, 99),
        "Insulin": (0, 600),
        "BMI": (10, 70),
        "DiabetesPedigreeFunction": (0, 2),
        "Age": (1, 120),
    }

    def validate_limits(data: dict) -> str:
        """Check if input data is within the defined limits."""
        for feature, (lower, upper) in feature_limits.items():
            if data[feature] < lower or data[feature] > upper:
                return f"Dữ liệu không thực tế {feature}: {data[feature]}"
        return "valid"

    @app.get('/')
    def welcome():
        return {"Welcome": "User"}

    @app.post('/predict/')
    async def predict_diabetes(input_data: Diabetes):
        try:
            # Prepare input data as a dictionary
            input_dict = {
                'Pregnancies': int(input_data.Pregnancies),
                'Glucose': int(input_data.Glucose),
                'BloodPressure': int(input_data.BloodPressure),
                'SkinThickness': int(input_data.SkinThickness),
                'Insulin': float(input_data.Insulin),
                'BMI': float(input_data.BMI),
                'DiabetesPedigreeFunction': float(input_data.DiabetesPedigreeFunction),
                'Age': int(input_data.Age)
            }

            # Validate input data against feature limits
            validation_result = validate_limits(input_dict)
            if validation_result != "valid":
                return {"prediction": validation_result}

            # Convert input data to a DataFrame
            x_values = pd.DataFrame([input_dict])

            # Make prediction
            prediction_proba = full_pipeline.predict_proba(x_values)[:, 1]
            prediction = (prediction_proba >= threshold).astype(int)
            # prediction = full_pipeline.predict(x_values)

            prediction_result = 'Diabetes' if prediction[0] == 1 else 'Not Diabetes'
            return {'prediction': prediction_result}

        except Exception as e:
            raise HTTPException(status_code=400, detail="API server is closed")

    return app


if __name__ == '__main__':
    app = API()  # Initialize app
    if app is not None:
        uvicorn.run(app, host='127.0.0.1', port=8000)
    else:
        print("API chạy thất bại")
