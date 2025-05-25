from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]
    classes = ['Setosa', 'Versicolor', 'Virginica']
    result = classes[prediction]
    return render_template('index.html', prediction_text=f'Predicted Iris Species: {result}')

if __name__ == '__main__':
    app.run(debug=True)
