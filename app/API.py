import uvicorn
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware # Post request for HTML
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

    # Must be rf or logr according to the project
    name = utilities.model_select()

    # Check whether the model exist or not
    check = bool(utilities.check_model(name))
    if check is False:
        return None

    full_pipeline = utilities.joblib_load(f'{name}')

    @app.post('/predict/')
    async def predict_adclick(input_data: Diabetes):
        try:
            x_values = pd.DataFrame({
                'Pregnancies': [int(input_data.Pregnancies)],
                'Glucose': [int(input_data.Glucose)],
                'BloodPressure': [int(input_data.BloodPressure)],
                'SkinThickness': [int(input_data.SkinThickness)],
                'Insulin': [int(input_data.Insulin)],
                'BMI': [float(input_data.BMI)],
                'DiabetesPedigreeFunction': [float(input_data.DiabetesPedigreeFunction)],
                'Age': [int(input_data.Age)]
            })

            prediction = full_pipeline.predict(x_values)

            prediction_result = 'Diabetes' if prediction[0] == 1 else 'Not Diabetes'
            return {'prediction': prediction_result}

        except Exception as e:
            raise HTTPException(status_code=400, detail="API server is closed")
    return app


if __name__ == '__main__':
    app = API() # Initialize app
    if app is not None:
        uvicorn.run(app,host='127:0:0:1',port=8000)
    else:
        print("API chạy thất bại")