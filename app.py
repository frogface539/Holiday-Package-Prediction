import streamlit as st
import pandas as pd
import joblib
import json

with open('model_features.json', 'r') as f:
    model_features = json.load(f)

model = joblib.load('rf_model.pkl')

st.set_page_config(page_title="Holiday Package Purchase Prediction", layout="centered")
st.title("Holiday Package Purchase Prediction")

st.write("Fill the customer details below to predict whether they are likely to purchase the package.")

def user_input_features():
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    duration = st.number_input('Number of Followups', min_value=0, value=2)
    income = st.number_input('Annual Income (in Lakhs)', min_value=0.0, value=5.0)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married'])

    data = {
        'Age': age,
        'NumberOfFollowups': duration,
        'AnnualIncome': income,
        'Gender_Female': 1 if gender == 'Female' else 0,
        'MaritalStatus_Single': 1 if marital_status == 'Single' else 0
    }
    return pd.DataFrame([data])

input_df = user_input_features()
input_df_encoded = pd.get_dummies(input_df)
input_df_encoded = input_df_encoded.reindex(columns=model_features, fill_value=0)

if st.button("Predict"):
    try:
        prediction = model.predict(input_df_encoded)[0]
        prob = model.predict_proba(input_df_encoded)[0][1]

        if prediction == 1:
            st.success(f"Likely to purchase the package (Confidence: {prob:.2f})")
        else:
            st.warning(f"Unlikely to purchase the package (Confidence: {1 - prob:.2f})")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
