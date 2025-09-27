import streamlit as st
import pandas as pd
import pickle

# ------------------------------------------------
# Load model and encoders
# ------------------------------------------------
@st.cache_resource
def load_model_and_encoders():
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model_data["model"], model_data["features_names"], encoders

model, feature_names, encoders = load_model_and_encoders()

# ------------------------------------------------
# Streamlit App
# ------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Provide customer details to check if they are likely to churn.")

# ------------------------------------------------
# Collect Inputs
# ------------------------------------------------
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "Yes", "No"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

# ------------------------------------------------
# Prepare input DataFrame
# ------------------------------------------------
input_data = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

input_df = pd.DataFrame([input_data])

# Encode categorical features using saved encoders
for column, encoder in encoders.items():
    if column in input_df.columns:
        input_df[column] = encoder.transform(input_df[column])

# Reorder columns to match training
input_df = input_df[feature_names]

# ------------------------------------------------
# Prediction
# ------------------------------------------------
if st.sidebar.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ The customer is **likely to churn**. (Churn probability: {probability:.2%})")
    else:
        st.success(f"✅ The customer is **not likely to churn**. (Churn probability: {probability:.2%})")

    st.write("### Input Summary")
    st.dataframe(input_df)
