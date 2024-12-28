# %%
import joblib
from azureml.core.model import Model
from flask import Flask, request, jsonify

app = Flask(__name__)

def init():
    global model
    model_path = Model.get_model_path('iris_model')
    model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    prediction = model.predict(data)
    return jsonify({'prediction': prediction.tolist()})
# %%
