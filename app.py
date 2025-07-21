# app.py
import streamlit as st
import pandas as pd
import joblib
import gzip
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress version warning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

try:
    with gzip.open("best_model.pkl.gz", "rb") as f:
        model = joblib.load(f)
    feature_names = joblib.load("feature_names.pkl")
except Exception as e:
    st.error(f"Error loading model or feature names: {e}")
    st.stop()

st.set_page_config(page_title="Employee Salary Estimator", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    .stSelectbox label, .stNumberInput label {
        color: pink !important;
    }
    .main-title {
        color: #00ffcc;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Employee Salary Estimator</div>', unsafe_allow_html=True)
st.markdown("Upload employee CSV file or fill in details below:")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Manual input fallback
if not uploaded_file:
    age = st.selectbox("Age", list(range(18, 61)))
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov'])
    education_num = st.selectbox("Education Number", [9, 10, 13, 14])  # numeric form
    marital = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced'])
    occupation = st.selectbox("Occupation", [
        'Exec-managerial', 'Craft-repair', 'Sales', 'Prof-specialty',
        'Other-service', 'Adm-clerical', 'Machine-op-inspct', 'Transport-moving',
        'Handlers-cleaners', 'Tech-support', 'Protective-serv', 'Farming-fishing',
        'Priv-house-serv', 'Armed-Forces'
    ])
    relationship = st.selectbox("Relationship", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried'])
    race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
    capital_loss = st.number_input("Capital Loss", 0, 99999, 0)
    hours = st.selectbox("Hours per Week", list(range(1, 101)))
    fnlwgt = 100000  # dummy value or input if required
    native_country = st.selectbox("Native Country", ['United-States', 'India', 'Mexico', 'Philippines', 'Other'])

    input_df = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'educational-num': education_num,
        'marital-status': marital,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours,
        'native-country': native_country
    }])
else:
    input_df = pd.read_csv(uploaded_file)

    # Drop "education" column if present
    if 'education' in input_df.columns:
        input_df = input_df.drop(columns=['education'])

# Predict
if st.button("Predict"):
    try:
        input_df = input_df.reindex(columns=feature_names)
        prediction = model.predict(input_df)
        st.success(f"Prediction: {prediction[0]}")

        # Chart
        if 'gender' in input_df.columns:
            st.subheader("📊 Gender Distribution")
            fig, ax = plt.subplots()
            input_df['gender'].value_counts().plot(kind='bar', ax=ax, color=['#e84393', '#0984e3'])
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
