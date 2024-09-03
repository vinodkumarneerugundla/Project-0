#importing necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.responses import JSONResponse, FileResponse

# Load the trained model
model = joblib.load('bike_price_model.pkl')

app = FastAPI()

class BikeData(BaseModel):
    model_name: int
    model_year: int
    kms_driven: float
    owner: int
    mileage: float
    power: float

@app.post("/predict/")
def predict_price(data: BikeData):
    # Prepare input data for prediction
    input_data = pd.DataFrame([[data.model_name, data.model_year, data.kms_driven, data.owner, data.mileage, data.power]],
                              columns=['model_name', 'model_year', 'kms_driven', 'owner', 'mileage', 'power'])
    # Predict the price
    prediction = model.predict(input_data)
    predicted_price = prediction[0]

    # Prepare response
    response = {
        "predicted_price": predicted_price,
        "plots_url": "http://127.0.0.1:8000/plots/"
    }

    return JSONResponse(content=response)

@app.get("/plots/")
def get_plots():
    # Return the saved plot image
    return FileResponse('plots.png')
