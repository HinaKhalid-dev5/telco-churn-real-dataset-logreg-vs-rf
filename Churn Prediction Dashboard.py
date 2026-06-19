import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── PAGE CONFIGURATION ───────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ── LOAD AND PREPARE DATA ────────────────────────────────
@st.cache_data
def load_and_train():
    # Load dataset
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Clean data
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    
    # Feature engineering
    service_cols = ['PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 
                    'StreamingMovies']
    df['TotalServices'] = (df[service_cols] == 'Yes').sum(axis=1)
    df['AvgMonthlySpend'] = df['TotalCharges'] / df['tenure']
    support_cols = ['TechSupport', 'OnlineSecurity', 
                    'DeviceProtection']
    df['SupportCount'] = (df[support_cols] == 'Yes').sum(axis=1)
    df['SpendTrend'] = df['MonthlyCharges'] - df['AvgMonthlySpend']
    df['TenureGroup'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['New', 'Developing', 'Established', 'Loyal'])
    
    # Encode categorical columns
    le = LabelEncoder()
    categorical_cols = ['gender', 'Partner', 'Dependents',
                        'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV',
                        'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod',
                        'TenureGroup']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    df['Churn'] = le.fit_transform(df['Churn'])
    
    # Train model
    X = df.drop(['customerID', 'Churn', 'Churn_numeric'
                 if 'Churn_numeric' in df.columns else 'Churn'],
                axis=1, errors='ignore')
    X = df.drop(['customerID', 'Churn'], axis=1, errors='ignore')
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train.columns.tolist()

model, feature_cols = load_and_train()

# ── HEADER ───────────────────────────────────────────────
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Enter customer details below to predict "
            "churn probability and risk category.")
st.divider()

# ── INPUT FORM ───────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Basic Information")
    tenure = st.slider("Tenure (months)", 1, 72, 12)
    monthly_charges = st.slider(
        "Monthly Charges ($)", 18.0, 120.0, 65.0)
    contract = st.selectbox(
        "Contract Type",
        ['Month-to-month', 'One year', 'Two year'])
    internet = st.selectbox(
        "Internet Service",
        ['DSL', 'Fiber optic', 'No'])
    payment = st.selectbox(
        "Payment Method",
        ['Electronic check', 'Mailed check',
         'Bank transfer (automatic)',
         'Credit card (automatic)'])

with col2:
    st.subheader("👤 Customer Profile")
    gender = st.selectbox("Gender", ['Male', 'Female'])
    senior = st.selectbox(
        "Senior Citizen", ['No', 'Yes'])
    partner = st.selectbox("Has Partner", ['Yes', 'No'])
    dependents = st.selectbox("Has Dependents", ['Yes', 'No'])
    paperless = st.selectbox(
        "Paperless Billing", ['Yes', 'No'])

with col3:
    st.subheader("📡 Services")
    phone = st.selectbox("Phone Service", ['Yes', 'No'])
    multiple_lines = st.selectbox(
        "Multiple Lines", ['Yes', 'No'])
    security = st.selectbox("Online Security", ['Yes', 'No'])
    backup = st.selectbox("Online Backup", ['Yes', 'No'])
    device = st.selectbox("Device Protection", ['Yes', 'No'])
    tech = st.selectbox("Tech Support", ['Yes', 'No'])
    tv = st.selectbox("Streaming TV", ['Yes', 'No'])
    movies = st.selectbox("Streaming Movies", ['Yes', 'No'])

st.divider()

# ── PREDICT BUTTON ───────────────────────────────────────
predict_clicked = st.button(
    "🔍 PREDICT CHURN", 
    type="primary",
    use_container_width=True)

if predict_clicked:
    # Build customer dictionary
    customer = {
        'gender': 1 if gender == 'Male' else 0,
        'SeniorCitizen': 1 if senior == 'Yes' else 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone == 'Yes' else 0,
        'MultipleLines': 1 if multiple_lines == 'Yes' else 0,
        'InternetService': ['DSL', 'Fiber optic',
                            'No'].index(internet),
        'OnlineSecurity': 1 if security == 'Yes' else 0,
        'OnlineBackup': 1 if backup == 'Yes' else 0,
        'DeviceProtection': 1 if device == 'Yes' else 0,
        'TechSupport': 1 if tech == 'Yes' else 0,
        'StreamingTV': 1 if tv == 'Yes' else 0,
        'StreamingMovies': 1 if movies == 'Yes' else 0,
        'Contract': ['Month-to-month', 'One year',
                     'Two year'].index(contract),
        'PaperlessBilling': 1 if paperless == 'Yes' else 0,
        'PaymentMethod': ['Electronic check',
                          'Mailed check',
                          'Bank transfer (automatic)',
                          'Credit card (automatic)'
                          ].index(payment),
        'MonthlyCharges': monthly_charges,
        'TotalCharges': monthly_charges * tenure,
    }

    # Engineered features
    services = ['PhoneService', 'MultipleLines',
                'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']
    customer['TotalServices'] = sum([
        customer[s] for s in services])
    customer['AvgMonthlySpend'] = (
        customer['TotalCharges'] / tenure
        if tenure > 0 else monthly_charges)
    customer['SupportCount'] = (
        customer['OnlineSecurity'] +
        customer['TechSupport'] +
        customer['DeviceProtection'])
    customer['SpendTrend'] = (
        monthly_charges - customer['AvgMonthlySpend'])
    if tenure <= 12:
        customer['TenureGroup'] = 0
    elif tenure <= 24:
        customer['TenureGroup'] = 1
    elif tenure <= 48:
        customer['TenureGroup'] = 2
    else:
        customer['TenureGroup'] = 3

    # Predict
    customer_df = pd.DataFrame([customer])
    customer_df = customer_df[feature_cols]
    churn_prob = model.predict_proba(customer_df)[0][1]

    # Risk category
    if churn_prob < 0.3:
        risk = 'Low Risk'
        risk_color = 'green'
        risk_emoji = '✅'
    elif churn_prob < 0.7:
        risk = 'Medium Risk'
        risk_color = 'orange'
        risk_emoji = '⚠️'
    else:
        risk = 'High Risk'
        risk_color = 'red'
        risk_emoji = '🚨'

    # Value tier
    if tenure <= 12 or monthly_charges < 35:
        value = 'Low Value'
        value_emoji = '🔴'
    elif tenure >= 49 and monthly_charges > 65:
        value = 'High Value'
        value_emoji = '🟢'
    else:
        value = 'Medium Value'
        value_emoji = '🟡'

    # Action
    if risk == 'High Risk':
        action = 'URGENT: Call immediately with retention offer'
    elif risk == 'Medium Risk':
        action = 'MONITOR: Add to retention watchlist'
    else:
        action = 'STABLE: Customer likely to stay'

    # ── RESULTS ──────────────────────────────────────────
    st.subheader("📈 Prediction Results")
    
    res1, res2, res3 = st.columns(3)
    
    with res1:
        st.metric(
            label="Churn Probability",
            value=f"{churn_prob*100:.1f}%",
            delta=None)
        st.progress(churn_prob)

    with res2:
        st.metric(
            label="Risk Category",
            value=f"{risk_emoji} {risk}")

    with res3:
        st.metric(
            label="Customer Value",
            value=f"{value_emoji} {value}")

    st.divider()
    
    # Action box
    if risk == 'High Risk':
        st.error(f"🚨 {action}")
    elif risk == 'Medium Risk':
        st.warning(f"⚠️ {action}")
    else:
        st.success(f"✅ {action}")

    # Customer summary
    st.subheader("📋 Customer Summary")
    summary_data = {
        'Feature': ['Tenure', 'Monthly Charges', 
                    'Contract', 'Internet Service',
                    'Total Services', 'Support Count'],
        'Value': [f'{tenure} months', 
                  f'${monthly_charges}',
                  contract, internet,
                  customer['TotalServices'],
                  customer['SupportCount']]
    }
    st.table(pd.DataFrame(summary_data))
