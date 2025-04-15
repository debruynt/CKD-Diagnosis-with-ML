import streamlit as st
import numpy as np
import pickle
from train import MLPBinaryClassifier

with open("mlp_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Chronic Kidney Disease Classifier")

st.write("Enter patient features below:")

features = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
            'abnormal_red_blood_cells', 'abnormal_pus_cell', 'pus_cell_clumps',
            'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine',
            'sodium', 'potassium', 'haemoglobin', 'packed_cell_volume', 
            'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 
            'diabetes_mellitus', 'coronary_artery_disease', 'poor_appetite', 
            'pedal_edema', 'anemia']

X_input = []
for feature in features:
    val = st.number_input(feature, value=0.0)
    X_input.append(val)

if st.button("Predict"):
    X = np.array(X_input).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0][0]
    result = "CKD (1)" if pred == 1 else "Not CKD (0)"
    st.success(f"Prediction: {result}")
