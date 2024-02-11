from flask import Flask, request, render_template
import numpy as np
import random

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    # temp_val = np.random.randint([0,1])
    # print(temp_val)
    val = [0,1]
    temp_val = random.choice(val)
    print(temp_val)
    if temp_val == 0:
        prediction = "Fraud"
    else:
        prediction = "Not Fraud"

    return render_template('index.html', prediction_text='Transaction Status: {}'.format(prediction))

if __name__ == "__main__":
    app.run()