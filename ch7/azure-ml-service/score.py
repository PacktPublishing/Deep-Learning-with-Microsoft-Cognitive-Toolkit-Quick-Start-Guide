import os
import json
import numpy as np
from azureml.core.model import Model
import onnxruntime


model = None


def init():
    global model
    
    model_path = Model.get_model_path('classify_flowers')
    model = onnxruntime.InferenceSession(model_path)
    

def run(raw_data):
    data = json.loads(raw_data)
    data = np.array(data).astype(np.float32)
    
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    prediction = model.run([output_name], { input_name: data})
    
    # Select the first output from the ONNX model.
    # Then select the first row from the returned numpy array.
    prediction = prediction[0][0]

    return json.dumps({'scores': prediction.tolist() })
