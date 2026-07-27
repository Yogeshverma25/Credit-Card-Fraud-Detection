import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration
st.set_page_config(page_title="Pro Fraud Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ Advanced Credit Card Fraud Detection System")
st.markdown("Equipped with **Explainable AI (XAI)**, **Live Model Comparison**, and **Transaction History**.")

# 2. Load Models and Data
@st.cache_resource
def load_rf_model():
    return joblib.load('fraud_model.pkl')

@st.cache_resource
def load_lr_model():
    return joblib.load('logistic_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('creditcard.csv')

rf_model = load_rf_model()
lr_model = load_lr_model()
df = load_data()

# 3. Session State (Memory for Auto-Fill and History)
if 'current_transaction' not in st.session_state:
    st.session_state.current_transaction = np.zeros(30)
if 'history' not in st.session_state:
    st.session_state.history = [] # This will store our mini-statement

# 4. Sidebar Controls
st.sidebar.header("1. Model Selection ⚖️")
model_choice = st.sidebar.selectbox("Choose AI Engine:", ["Random Forest (High Accuracy)", "Logistic Regression (Fast)"])
active_model = rf_model if "Random" in model_choice else lr_model

st.sidebar.header("2. Auto-Fill Simulator 🤖")
if st.sidebar.button("🟢 Simulate Normal Transaction"):
    sample = df[df['Class'] == 0].sample(1).drop('Class', axis=1).values[0]
    st.session_state.current_transaction = sample

if st.sidebar.button("🔴 Simulate Fraud Transaction"):
    sample = df[df['Class'] == 1].sample(1).drop('Class', axis=1).values[0]
    st.session_state.current_transaction = sample

st.sidebar.header("3. Manual Tweaks 🎛️")
amount = st.sidebar.number_input("Amount ($)", value=float(st.session_state.current_transaction[29]))
v1 = st.sidebar.slider("V1 (PCA)", -50.0, 50.0, float(st.session_state.current_transaction[1]))
v14 = st.sidebar.slider("V14 (PCA)", -50.0, 50.0, float(st.session_state.current_transaction[14]))

# Prepare final input data
input_data = st.session_state.current_transaction.copy()
input_data[29] = amount
input_data[1] = v1
input_data[14] = v14
columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
input_df = pd.DataFrame([input_data], columns=columns)

# 5. Main Dashboard Area
st.subheader("Current Transaction Data")
st.dataframe(input_df)

if st.button("Run Fraud Detection", type="primary"):
    with st.spinner('Analyzing...'):
        # Prediction
        prob = active_model.predict_proba(input_df)[0][1]
        status = "Fraud" if prob > 0.50 else "Approved"
        
        # UI Results
        st.write(f"### Risk Score (via {model_choice}):")
        st.progress(float(prob))
        st.write(f"**{prob * 100:.2f}% chance of being fraudulent.**")
        
        if prob > 0.50:
            st.error("🚨 **FRAUD DETECTED!** Transaction Blocked.")
        elif prob > 0.20:
            st.warning("⚠️ **WARNING:** Unusual patterns detected.")
        else:
            st.success("✅ **TRANSACTION APPROVED.**")

        # --- FEATURE 1: EXPLAINABLE AI (XAI) ---
        if "Random" in model_choice:
            st.markdown("### 🧠 Explainable AI: Why did the model make this decision?")
            st.write("The chart below shows which features heavily influenced the model's brain:")
            
            # Get feature importance from Random Forest
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[-5:] # Top 5 features
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(range(len(indices)), importances[indices], color='skyblue')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([columns[i] for i in indices])
            ax.set_title("Top 5 Decision Factors")
            st.pyplot(fig)

        # --- FEATURE 3: TRANSACTION HISTORY ---
        # Save this run to our history memory
        st.session_state.history.append({
            "Engine Used": model_choice,
            "Amount": f"${amount:.2f}",
            "Risk Score": f"{prob*100:.2f}%",
            "Final Decision": status
        })

# --- MINI-STATEMENT UI ---
st.markdown("---")
st.subheader("📜 Recent Transactions Log")
if len(st.session_state.history) > 0:
    # Reverse the list so the newest transaction is at the top
    history_df = pd.DataFrame(st.session_state.history[::-1])
    
    # Simple color formatting function for the table
    def color_status(val):
        color = 'red' if val == 'Fraud' else 'green'
        return f'color: {color}; font-weight: bold'
    
    st.dataframe(history_df.style.applymap(color_status, subset=['Final Decision']), use_container_width=True)
else:
    st.write("No transactions have been analyzed yet during this session.")