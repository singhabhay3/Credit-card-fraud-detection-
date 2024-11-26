import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Bank Fraud Detection System",
    layout="wide",
    page_icon="💰",
    initial_sidebar_state="expanded",
)

# Styling for a professional look
st.markdown(
    """
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f7f7f7;
    }
    .css-18e3th9 {
        background-color: #1e3a8a !important; 
        color: #ffffff !important;
    }
    .css-1v0mbdj, .css-1cxoyoj {
        background-color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# User credentials for login
users = {"bankadmin": "secure123", "financialanalyst": "analytics2023"}

# Initialize session state variables
if "data" not in st.session_state:
    st.session_state["data"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "scaler" not in st.session_state:
    st.session_state["scaler"] = None

# Sidebar: Login Interface
st.sidebar.image("bank_login.png", width=300)  # Replace with your bank logo
st.sidebar.title("💼 Banking Portal")
st.sidebar.header("User Login")

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if login_button:
    if username in users and users[username] == password:
        st.session_state["logged_in"] = True
        st.sidebar.success(f"Welcome, {username}!")
    else:
        st.sidebar.error("Invalid credentials. Please try again.")

if st.session_state["logged_in"]:
    # Sidebar Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Dashboard", "Upload Data", "Model Training", "Transaction Prediction"])

    # Dashboard
    if page == "Dashboard":
        st.title("🏦 Fraud Detection Dashboard")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(
                """
                **Overview:**
                This system helps banks to:
                - Detect fraudulent transactions with high precision.
                - Visualize key metrics for decision-making.
                - Perform real-time fraud detection.
                - Prediction** for individual transactions.
                - High Accuracy Fraud Detection** using Logistic Regression.
                """
            )
        with col2:
            st.image("bank_dashboard.png", use_column_width=True)  # Add a banking-related dashboard image

    # Upload Data
    elif page == "Upload Data":
        st.title("📂 Upload Fraud Detection Dataset")
        uploaded_file = st.file_uploader("Upload a CSV file (e.g., credit card data):", type="csv")
        if uploaded_file:
            try:
                st.session_state["data"] = pd.read_csv(uploaded_file)
                st.write("### Uploaded Data Preview")
                st.dataframe(st.session_state["data"].head(), use_container_width=True)
                st.success("Dataset uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    # Model Training
    elif page == "Model Training":
        st.title("📊 Model Training & Evaluation")

        # Use default dataset if no upload
        if st.session_state["data"] is None:
            st.warning("No dataset uploaded. Using default dataset.")
            st.session_state["data"] = pd.read_csv("creditcard.csv")

        data = st.session_state["data"]
        legit = data[data.Class == 0]
        fraud = data[data.Class == 1]

        # Display Dataset Stats
        st.write("### Dataset Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Legitimate Transactions")
            st.write(legit.describe())
        with col2:
            st.write("#### Fraudulent Transactions")
            st.write(fraud.describe())

        # Balance Dataset
        legit_sample = legit.sample(n=len(fraud), random_state=2)
        balanced_data = pd.concat([legit_sample, fraud], axis=0)

        # Scale Features
        scaler = StandardScaler()
        X = balanced_data.drop(columns="Class", axis=1)
        y = balanced_data["Class"]
        X = scaler.fit_transform(X)
        st.session_state["scaler"] = scaler

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

        # Train Logistic Regression Model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        st.session_state["model"] = model

        # Evaluate Model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        test_acc = accuracy_score(y_test, y_pred)

        # Metrics
        st.write("### Model Performance Metrics")
        st.metric("Test Accuracy", f"{test_acc * 100:.2f}%")
        st.write("#### Classification Report")
        st.text(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # ROC Curve
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("Receiver Operating Characteristic")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        st.pyplot()

    # Transaction Prediction
    elif page == "Transaction Prediction":
        st.title("🔍 Transaction Fraud Prediction")
        if st.session_state["model"] is None:
            st.warning("Train the model first!")
        else:
            st.write("### Enter Transaction Details")
            data = st.session_state["data"]
            input_features = []
            for col in data.drop(columns="Class").columns:
                value = st.number_input(f"{col}:", value=0.0, format="%.2f")
                input_features.append(value)

            if st.button("Predict"):
                input_features = np.array(input_features).reshape(1, -1)
                input_features = st.session_state["scaler"].transform(input_features)
                prediction = st.session_state["model"].predict(input_features)
                pred_prob = st.session_state["model"].predict_proba(input_features)

                if prediction[0] == 0:
                    st.success("✅ Legitimate Transaction")
                else:
                    st.error("🚨 Fraudulent Transaction")
                st.write(f"Legitimate Probability: {pred_prob[0][0] * 100:.2f}%")
                st.write(f"Fraudulent Probability: {pred_prob[0][1] * 100:.2f}%")
else:
    st.warning("Please log in to access the system.")
