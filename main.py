import os
import pickle
import pandas as pd

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from starter.ml.data import process_data


# Declare the data object with its components and their type
class Person(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to the prediction of salary from census data!"}

# Define a POST for the prediction of salary
@app.post("/predict")
async def prediction(person: Person):

    with open(os.path.join("model", "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join("model", "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    with open(os.path.join("model", "label_binarizer.pkl"), "rb") as f:
        label_binarizer = pickle.load(f)

    data = pd.DataFrame([person.dict()])
    X, _, _, _ = process_data(data, categorical_features=cat_features, label=None, training=False, encoder=encoder)

    pred = model.predict(X)

    if pred == 1:
        return {"the result": "Over 50k"}
    else:
        return {"the result": "Less 50k"}


    
    

