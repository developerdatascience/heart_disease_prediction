import json
import pickle

from flask import Flask, request, jsonify, url_for, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

#Load the model
logmodel = pickle.load(open("logisticmodel.pkl", "rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data =np.array(list(data.values())).reshape(1,-1)
    output = logmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict', methods=['GET','POST', ])
def predict():
    if request.method == "POST":
        data = [int(x) for x in request.form.values()]
        final_input=np.array(data).reshape(1, -1)
        print(final_input)
        output = logmodel.predict(final_input)[0]
        if output == 0:
            result = "Person is Safe"
        else:
            result = "Person is Not Safe"
    return render_template("home.html", prediction_text= result)


if __name__=="__main__":
    app.run(debug=True)