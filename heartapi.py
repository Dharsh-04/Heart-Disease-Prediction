from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model_path = r'C:\Users\Admin\Desktop\MLPRACTICE\Heart_disease_Predict_Project\heartmodel.pkl'
model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return render_template('heartWeb.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form input
    age = request.form['age']
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = request.form['oldpeak']
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']

    # Convert the inputs to a numpy array for prediction
    input_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], dtype=float)

    # Predict using the loaded model
    prediction = model.predict(input_features)[0]

    # Interpret the result
    if prediction == 1:
        pred = "Heart Disease Found"
    else:
        pred = "No Heart Disease Found"

    return render_template('heartWeb.html', pred=pred)

if __name__ == '__main__':
    app.run(debug=True)
