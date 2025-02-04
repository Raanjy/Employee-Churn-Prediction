# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pickle
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")


file_path = os.path.join("C:\ML\Employee Churn", "Final_model.pkl")

    
# Load the model
with open("C:\ML\Employee Churn\Final_model.pkl", 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def predict(input_data):
    input_array = np.array(input_data).reshape(1, -1)  # Reshape for model input
    prediction = model.predict(input_array)
    return prediction[0]

# Streamlit UI
st.title("Machine Learning Model Deployment")

# Input fields (Modify according to your model's features)
satisfaction_level = st.number_input("satisfaction_level")
number_project = st.number_input("number_project")
last_evaluation = st.number_input("last_evaluation")
average_montly_hours = st.number_input("average_montly_hours")
time_spend_company = st.number_input("time_spend_company")
Work_accident = st.number_input("Work_accident")
promotion_last_5years = st.number_input("promotion_last_5years")
Departments__IT = st.number_input("Departments__IT")
Departments__RandD = st.number_input("Departments__RandD")
Departments__hr = st.number_input("Departments__hr")
Departments__accounting = st.number_input("Departments__accounting")
Departments__management = st.number_input("Departments__management")
Departments__marketing = st.number_input("Departments__marketing")
Departments__product_mng = st.number_input("Departments__product_mng")
Departments__sales = st.number_input("Departments__sales")
Departments__support = st.number_input("Departments__support")
Departments__technical = st.number_input("Departments__technical")
salary_high = st.number_input("salary_high")
salary_low = st.number_input("salary_low")
salary_medium = st.number_input("salary_medium")

# Button to make predictions
if st.button("Predict"):
    input_data = [satisfaction_level,number_project, last_evaluation, average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,Departments__IT,Departments__RandD,Departments__accounting,Departments__hr,Departments__management,Departments__marketing,Departments__product_mng,Departments__sales,Departments__support,Departments__technical,salary_high,salary_low,salary_medium]  # Modify based on your model
    result = predict(input_data)
    st.write(f"Prediction: {result}")
    
# Display the result
    if result == 1:
        st.write("Prediction: **Churn**")
    else:
        st.write("Prediction: **No Churn**")

