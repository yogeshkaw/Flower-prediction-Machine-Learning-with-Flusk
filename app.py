from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

mul_reg = open("model.pkl", "rb")
ml_model = joblib.load(mul_reg)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        #print(request.form.get('sepal_length'))
        try:
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])
            pred_args = [sepal_length, sepal_width, petal_length, petal_width]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            model_prediction1 = ml_model.predict(pred_args_arr)
            if model_prediction1==0:
                model_prediction='setosa'
            elif model_prediction1==1:
                model_prediction='versicolor'
            else:
                model_prediction='virginica'
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction = model_prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
