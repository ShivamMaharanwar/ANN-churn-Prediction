import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OrdinalEncoder
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('salary_regression_model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
    
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
    
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
# Streamlit app
st.title("Salary Prediction App")
st.write("Enter the details to predict the salary:")

# Input fields
geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,60,30)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',('Yes','No'))
is_active_member=st.selectbox('Is Active Member',('Yes','No'))

# prepare the input data
input_data = pd.DataFrame({
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Balance': [balance],
    'CreditScore': [credit_score],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card=='Yes' else 0],
    'IsActiveMember': [1 if is_active_member=='Yes' else 0]
})

# Encode categorical variables
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.drop('Geography', axis=1), geo_df], axis=1)

# Ensure input has the same features (and order) the scaler was fitted with
# add any missing columns with default 0, then reorder to match scaler.feature_names_in_
feature_names = getattr(scaler, 'feature_names_in_', None)
if feature_names is not None:
    for feat in feature_names:
        if feat not in input_data.columns:
            input_data[feat] = 0
    input_data = input_data[list(feature_names)]

# Scale the input data (use numpy array to avoid DataFrame feature-name checks)
input_data_scaled = scaler.transform(input_data.values.astype(float))

if st.button('Predict Salary'):
    # Make prediction
    predicted_salary = model.predict(input_data_scaled)
    st.success(f'The predicted salary is: ${predicted_salary[0][0]:.2f}')