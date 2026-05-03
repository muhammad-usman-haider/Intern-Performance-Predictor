import streamlit as st
import numpy as np
import joblib

# Load trained model (choose RF or XGB depending on what you saved)
rf_model = joblib.load("models/rf_best.pkl")
xgb_model = joblib.load("models/xgb_best.pkl")

st.title("🎓 Intern Performance Predictor")

st.write("Enter the intern's metrics below to predict performance:")

# Input fields
attendance_rate = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
task_completion_rate = st.number_input("Task Completion Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
avg_feedback_score = st.number_input("Average Feedback Score (0–10)", min_value=0.0, max_value=10.0, step=0.1)
final_assessment_score = st.number_input("Final Assessment Score (0–100)", min_value=0.0, max_value=100.0, step=0.1)

# Choose model
model_choice = st.radio("Select Model", ("Random Forest", "XGBoost"))

if st.button("Predict Performance"):
    # Prepare input
    features = np.array([[attendance_rate, task_completion_rate, avg_feedback_score, final_assessment_score]])
    
    if model_choice == "Random Forest":
        prediction = rf_model.predict(features)[0]
    else:
        prediction = xgb_model.predict(features)[0]
    
    st.success(f"Predicted Performance Label: {prediction}")