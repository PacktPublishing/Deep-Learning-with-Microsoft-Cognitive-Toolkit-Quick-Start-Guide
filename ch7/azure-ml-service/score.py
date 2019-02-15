import pandas as pd
import json
import os
from azureml.core.model import Model
from cntk import load_model

model = None


def init():
    global model
    
    model_path = Model.get_model_path('classify_flowers')
    model = load_model(model_path)
    

def run(raw_data):
    data = pd.read_json(raw_data).values.reshape(1,-1)
    prediction = model(data.values)[0]
    
    return json.dumps({'scores': prediction.tolist() })