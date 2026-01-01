import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


## load the trained model
model= tf.keras.models.load_model('ann_churn_model.h5')

## load the encoders and scalers
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
    
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
    
    
## Streamlit app
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")

## Input fields
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender=st.selectbox("Gender", label_encoder_gender.classes_)
age=st.slider("Age", 18, 100, 30)
balance=st.number_input("Balance")
credit_score=st.number_input("CreditScore")
estimated_salary=st.number_input("Estimated Salary")
tenure=st.slider("Tenure (years)",0,10)
num_of_products=st.slider("Number of Products",1,4)
has_cr_card=st.selectbox("Has Credit Card", [0, 1])
is_active_member=st.selectbox("Is Active Member", [0, 1])

## prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data, geo_df], axis=1)
input_data = input_data.drop(columns=['Geography'], errors='ignore')
input_data_scaled = scaler.transform(input_data)

#predict churn probability
if st.button("Predict Churn Probability"):
    churn_prob = model.predict(input_data_scaled)
    st.write(f"Churn Probability: {churn_prob[0][0]:.2f}")
    
    if churn_prob[0][0] > 0.5:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is unlikely to churn.")