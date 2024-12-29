# %%
import joblib
from flask import Flask, request, jsonify
from azure.ai.ml.entities import Model

app = Flask(__name__)

def init():
    global model
    model_path = Model.get_model_path('iris_model')
    model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def run():
    data = request.json['data']
    prediction = model.predict(data)
    return jsonify({'prediction': prediction.tolist()})
# %%
