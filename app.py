# app.py

from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model from file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['Gender']
    if gender == "Male":
        Gender = 1
    elif gender == 'Female':
        Gender = 0
    estimated_salary = int(request.form['EstimatedSalary'])
    age = int(request.form['Age'])

    # Prepare feature array
    feature = np.array([[Gender, age, estimated_salary]])

    # Make prediction
    prediction = model.predict(feature)

    # Convert prediction to human-readable format
    pred_res = "Will Purchase" if prediction[0] == 1 else "Will Not Purchase"

    return render_template('result.html', pred_res=pred_res)


if __name__ == '__main__':
    app.run(debug=True)
