import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu


# Training the Modles

diabetes_model = pickle.load(open( "Diabetes_model.sv","rb"))
heart_model = pickle.load(open("Heartdisease_model_v2.sav","rb"))
parkinson_model = pickle.load(open("parkinsons_model.sav","rb"))

#  Sidebar navigation 
with st.sidebar:
    selected = option_menu(
        menu_title="Multiple Disease Prediction",
        options=[
            "Diabetes Prediction",
            "Heart Disease Prediction",
            "Parkinsons Prediction"
        ],
        icons=["activity", "heart", "person"],
    )


#  Diabetes Prediction Page

if selected == ("Diabetes Prediction"):
    st.title("Diabetes Prediction using ML")

    c1, c2, c3 = st.columns(3)

    with c1:
        pregnancies = st.text_input("Number of Pregnancies")
        skin_thickness = st.text_input("Skin Thickness value")
        diabetes_pedigree = st.text_input("Diabetes Pedigree Function")

    with c2:
        glucose = st.text_input("Glucose Level")
        insulin = st.text_input("Insulin Level")
        age = st.text_input("Age")

    with c3:
        blood_pressure = st.text_input("Blood Pressure value")
        bmi = st.text_input("BMI value")
        st.write("")          

    diab_result = ""
    
    
    if st.button("Diabetes Test Result"):
        try:
            # Convert to float and build feature vector
            features = np.asarray([
                float(pregnancies), float(glucose), float(blood_pressure),
                float(skin_thickness), float(insulin), float(bmi),
                float(diabetes_pedigree), float(age)
            ]).reshape(1, -1)

            prediction = diabetes_model.predict(features)
            diab_result = (
                "The person **is diabetic**." if prediction[0] == 1
                else "The person **is not diabetic**."
            )
        except ValueError:
            diab_result = "Please enter only numerical values."

    if diab_result:
        st.success(diab_result)

#  Heart Disease Prediction Page


elif selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")

    c1, c2, c3 = st.columns(3)

    # Column 1
    with c1:
        age = st.text_input("Age",)
        trestbps = st.text_input("Resting Blood Pressure (trestbps)")
        restecg = st.text_input("Resting ECG (restecg)")
        oldpeak = st.text_input("ST Depression (oldpeak)")
        thal = st.text_input("Thalassemia (thal)")

    # Column 2
    with c2:
        sex = st.text_input("Gender  (0 = F, 1 = M)")
        chol = st.text_input("Cholesterol")
        thalach = st.text_input("Max Heart Rate (thalach)")
        slope = st.text_input("Slope of ST segment")

    # Column 3
    with c3:
        cp = st.text_input("Chest Pain Type (cp)")
        fbs = st.text_input("Fasting Blood Sugar (fbs)")
        exang = st.text_input("Exercise‑Induced Angina (exang)")
        ca = st.text_input("Major Vessels Colored (ca)")
        st.write("")          

    heart_result = ""

    if st.button("Heart Disease Test Result"):
        try:
            features = np.asarray([
                float(age), float(sex), float(cp), float(trestbps), float(chol),
                float(fbs), float(restecg), float(thalach), float(exang),
                float(oldpeak), float(slope), float(ca), float(thal)
            ]).reshape(1, -1)

            prediction = heart_model.predict(features)
            heart_result = (
                "The person **has heart disease**."
                if prediction[0] == 1 else
                "The person **does not have heart disease**."
            )
        except ValueError:
            heart_result = ("please enter the numerical values.")

    st.success(heart_result)
    
 
# Parkinsons Predection page

elif selected == "Parkinsons Prediction":
    st.title("Parkinson’s Prediction using ML")

    
    cols = st.columns(5)
    labels = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
        "Spread1", "Spread2", "D2", "PPE"
    ]

    inputs = []
    for i, label in enumerate(labels):
        with cols[i % 5]:
            inputs.append(st.text_input(label, key=label))

    parkison_result = ""

    # Button to trigger prediction
    if st.button("Parkinson Test Result"):
        try:
            # Convert inputs to float
            input_data = [float(value) for value in inputs]

            # Make prediction using the trained model
            parkison_prediction = parkinson_model.predict([input_data])

            if parkison_prediction[0] == 1:
                parkison_result = "The person **has Parkinson’s disease**."
            else:
                parkison_result = "The person **does not have Parkinson’s disease**."
        except ValueError:
            st.error("Please enter valid numerical values in all fields.")

    # Show result
    if parkison_result:
        st.success(parkison_result)
    
    
    
    
    
