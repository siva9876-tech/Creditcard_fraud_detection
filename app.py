import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------------------
# Load the trained model
# ---------------------------------------
@st.cache_resource
def load_model():
    with open("rf.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ---------------------------------------
# Streamlit App UI
# ---------------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

st.title("💳 Credit Card Fraud Detection App")
st.markdown("Enter transaction details below to predict whether it’s **fraudulent or legitimate.**")

# ---------------------------------------
# User input form
# ---------------------------------------
st.sidebar.header("Input Transaction Features")

# Example feature inputs — adjust to match your dataset columns
# (Use the same feature names that were used when training the model)
V_features = [f"V{i}" for i in range(1, 29)]
inputs = {}

for v in V_features:
    inputs[v] = st.sidebar.number_input(f"{v}", value=0.0)

amount = st.sidebar.number_input("Transaction Amount", value=0.0)
inputs["Amount"] = amount

# Convert inputs to DataFrame
input_df = pd.DataFrame([inputs])

# ---------------------------------------
# Prediction
# ---------------------------------------
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        st.subheader("✅ Prediction Result")
        if prediction == 1:
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            st.success("💰 Legitimate Transaction")

        if probability is not None:
            st.write(f"**Fraud Probability:** {probability:.4f}")

        st.dataframe(input_df.T.rename(columns={0: "Value"}))
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# ---------------------------------------
# Footer
# ---------------------------------------
st.markdown("---")
st.caption("Developed with ❤️ using SIVA SHANKAR")

