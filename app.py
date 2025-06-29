import streamlit as st
import numpy as np
import joblib

# Load trained model and feature columns
model = joblib.load("logistic_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("IGBASAN ❤️ Heart Disease Risk Prediction App")

st.markdown("### Please fill in all the patient details below:")

# Collect all 13 inputs
age = st.slider("Age of the patient", 25, 100, 50)
sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.slider("Chest Pain Type (0–3)", 0, 3, 1)
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 7.0, 1.0, step=0.1)
slope = st.selectbox("Slope of the ST Segment (0–2)", [0, 1, 2])
ca = st.slider("Number of Major Vessels Colored (0–3)", 0, 3, 0)
thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: {
    1: "Normal",
    2: "Fixed Defect",
    3: "Reversible Defect"
}[x])

# Prediction
if st.button("Predict Heart Disease"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    
    prediction = model.predict(input_data)[0]
    result = "🩺 Heart Disease Detected" if prediction == 1 else "✅ No Heart Disease"
    
    st.subheader("Prediction Result:")
    st.success(result)
