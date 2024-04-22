# app.py

import warnings
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Suppress warning about inconsistent scikit-learn versions
warnings.filterwarnings("ignore", category=UserWarning)

# Load the trained model from file


def load_model():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model


try:
    model = load_model()
except ValueError as e:
    print("Error loading the model:", e)

# Function to preprocess input data


def preprocess_input(gender, age, estimated_salary):
    Gender = 1 if gender == "Male" else 0
    age = int(age)
    estimated_salary = int(estimated_salary)
    return np.array([[Gender, age, estimated_salary]])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['Gender']
    age = request.form['Age']
    estimated_salary = request.form['EstimatedSalary']

    # Preprocess input data
    feature = preprocess_input(gender, age, estimated_salary)

    if 'model' not in globals():
        return render_template('index.html', pred_res="Model loading failed. Please try again later.")

    # Make prediction
    try:
        prediction = model.predict(feature)
        # Convert prediction to human-readable format
        pred_res = "Will Purchase" if prediction[0] == 1 else "Will Not Purchase"
    except Exception as e:
        print("Error making prediction:", e)
        pred_res = "Prediction failed. Please try again later."

    return render_template('index.html', pred_res=pred_res)


if __name__ == '__main__':
    app.run(debug=True)
