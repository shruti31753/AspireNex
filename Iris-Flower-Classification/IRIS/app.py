from flask import Flask, render_template, request
import os
import pickle

app = Flask(__name__)

model_path = os.path.join(os.getcwd(), 'finalized_model.sav')

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: '{model_path}' not found. Please check the file path.")
    model = None  

@app.route('/')
def home():
    return render_template('index.html', result='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        if model:
            prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
        else:
            prediction = 'Model not loaded'

        return render_template('index.html', result=prediction)

    except Exception as e:
        return render_template('index.html', result='Error: Invalid input. Please check your input values.')

if __name__ == '__main__':
    app.run(debug=True)
