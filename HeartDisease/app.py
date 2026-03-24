from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# ✅ Load files
model = joblib.load("logistic_regression_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        raw_input = {
            'Age': int(data['age']),
            'RestingBP': int(data['resting_bp']),
            'Cholesterol': int(data['cholesterol']),
            'FastingBS': int(data['fasting_bs']),
            'MaxHR': int(data['max_hr']),
            'Oldpeak': float(data['oldpeak']),
            'Sex_' + data['sex']: 1,
            'ChestPainType_' + data['chest_pain']: 1,
            'RestingECG_' + data['resting_ecg']: 1,
            'ExerciseAngina_' + data['exercise_angina']: 1,
            'ST_Slope_' + data['st_slope']: 1
        }

        input_df = pd.DataFrame([raw_input])

        # Missing columns fill
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Correct order
        input_df = input_df[expected_columns]

        # Scale
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)[0]

        return jsonify({"result": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
    
print("Server started")