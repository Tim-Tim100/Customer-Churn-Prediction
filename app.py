"""
================================================================================
TELCO CHURN PREDICTION - STREAMLIT APP (FIXED PREPROCESSING)
================================================================================
Matches exact preprocessing from training
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Churn Risk Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔮 Customer Churn Risk Predictor")
st.markdown("**Predict if a customer will leave the company**")

# ============================================================================
# LOAD MODEL & SCALER
# ============================================================================

@st.cache_resource
def load_artifacts():
    try:
        # Load Gradient Boosting model
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_order = joblib.load('models/feature_order.pkl')
        return model, scaler, feature_order, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, feature_order, error = load_artifacts()

if error:
    st.error(f"❌ Could not load model: {error}")
    st.stop()

st.success("✅ Model loaded")

# ============================================================================
# USER INPUT
# ============================================================================

st.header("📋 Enter Customer Information")

tab1, tab2, tab3, tab4 = st.tabs(["👤 Personal", "📱 Services", "💰 Billing", "📅 Contract"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen (65+)?", ["No", "Yes"])
    with col2:
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        phone_service = st.selectbox("Phone Service?", ["No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    with col2:
        online_security = st.selectbox("Online Security?", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support?", ["No", "Yes"])
    
    col3, col4 = st.columns(2)
    with col3:
        online_backup = st.selectbox("Online Backup?", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection?", ["No", "Yes"])
    with col4:
        streaming_tv = st.selectbox("Streaming TV?", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies?", ["No", "Yes"])

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    with col2:
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)
    
    col3, col4 = st.columns(2)
    with col3:
        paperless_billing = st.selectbox("Paperless Billing?", ["No", "Yes"])
    with col4:
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col2:
        tenure = st.slider("Tenure (months)", 0, 72, 24)

# ============================================================================
# PREPROCESSING FUNCTION - EXACT MATCH TO TRAINING
# ============================================================================

def preprocess_input(
    gender, senior_citizen, partner, dependents,
    phone_service, internet_service, online_security, tech_support,
    online_backup, device_protection, streaming_tv, streaming_movies,
    monthly_charges, total_charges, paperless_billing, payment_method,
    contract, tenure, feature_order
):
    """
    Preprocess user input using EXACT feature order from training
    """
    try:
        # Create dataframe with ALL possible values
        data = {
            'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
            'Partner': [1 if partner == "Yes" else 0],
            'Dependents': [1 if dependents == "Yes" else 0],
            'tenure': [tenure],
            'PhoneService': [1 if phone_service == "Yes" else 0],
            'PaperlessBilling': [1 if paperless_billing == "Yes" else 0],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'TotalCharges_log': [np.log1p(total_charges)],
            'gender_Male': [1 if gender == "Male" else 0],
            'MultipleLines_No phone service': [1 if phone_service == "No" else 0],
            'MultipleLines_Yes': [1 if phone_service == "Yes" else 0],
            'InternetService_Fiber optic': [1 if internet_service == "Fiber optic" else 0],
            'InternetService_No': [1 if internet_service == "No" else 0],
            'OnlineSecurity_No internet service': [1 if internet_service == "No" else 0],
            'OnlineSecurity_Yes': [1 if online_security == "Yes" else 0],
            'OnlineBackup_No internet service': [1 if internet_service == "No" else 0],
            'OnlineBackup_Yes': [1 if online_backup == "Yes" else 0],
            'DeviceProtection_No internet service': [1 if internet_service == "No" else 0],
            'DeviceProtection_Yes': [1 if device_protection == "Yes" else 0],
            'TechSupport_No internet service': [1 if internet_service == "No" else 0],
            'TechSupport_Yes': [1 if tech_support == "Yes" else 0],
            'StreamingTV_No internet service': [1 if internet_service == "No" else 0],
            'StreamingTV_Yes': [1 if streaming_tv == "Yes" else 0],
            'StreamingMovies_No internet service': [1 if internet_service == "No" else 0],
            'StreamingMovies_Yes': [1 if streaming_movies == "Yes" else 0],
            'Contract_One year': [1 if contract == "One year" else 0],
            'Contract_Two year': [1 if contract == "Two year" else 0],
            'PaymentMethod_Credit card (automatic)': [1 if payment_method == "Credit card (automatic)" else 0],
            'PaymentMethod_Electronic check': [1 if payment_method == "Electronic check" else 0],
            'PaymentMethod_Mailed check': [1 if payment_method == "Mailed check" else 0],
            'tenure_0_6m': [1 if tenure <= 6 else 0],
            'tenure_6_12m': [1 if 6 < tenure <= 12 else 0],
            'tenure_1_2y': [1 if 12 < tenure <= 24 else 0],
            'tenure_2y_plus': [1 if tenure > 24 else 0],
        }
        
        df = pd.DataFrame(data)
        
        # Select features in EXACT order from training
        df = df[feature_order]
        
        return df, None
        
    except Exception as e:
        return None, str(e)

# ============================================================================
# PREDICTION
# ============================================================================

st.markdown("---")

if st.button("🔮 Predict Churn Risk", use_container_width=True):
    df_input, preprocess_error = preprocess_input(
        gender, senior_citizen, partner, dependents,
        phone_service, internet_service, online_security, tech_support,
        online_backup, device_protection, streaming_tv, streaming_movies,
        monthly_charges, total_charges, paperless_billing, payment_method,
        contract, tenure, feature_order
    )
    
    if preprocess_error:
        st.error(f"❌ Preprocessing Error: {preprocess_error}")
    else:
        try:
            # Scale
            df_scaled = scaler.transform(df_input)
            
            # Predict
            churn_prob = float(model.predict_proba(df_scaled)[0, 1])
            
            # ================================================================
            # DISPLAY RESULTS
            # ================================================================
            
            st.header("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Churn Risk", f"{churn_prob*100:.1f}%")
            
            with col2:
                if churn_prob < 0.33:
                    st.success("🟢 LOW RISK")
                elif churn_prob < 0.67:
                    st.warning("🟡 MEDIUM RISK")
                else:
                    st.error("🔴 HIGH RISK")
            
            with col3:
                if churn_prob > 0.6:
                    st.metric("Action", "⚡ URGENT")
                elif churn_prob > 0.33:
                    st.metric("Action", "👁️ MONITOR")
                else:
                    st.metric("Action", "✅ OK")
            
            # Risk Gauge
            st.subheader("📈 Risk Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                title={"text": "Churn Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 33], "color": "#90EE90"},
                        {"range": [33, 67], "color": "#FFD700"},
                        {"range": [67, 100], "color": "#FF6B6B"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommended Actions")
            
            if churn_prob > 0.6:
                st.error("🔴 **HIGH PRIORITY**")
                st.markdown("- 📞 Call within 24 hours\n- 💰 Offer 15-20% discount\n- 🎁 Service upgrade\n- 👨‍💼 Dedicated support")
            elif churn_prob > 0.33:
                st.warning("🟡 **MEDIUM PRIORITY**")
                st.markdown("- 📧 Send personalized offer\n- 💬 Check-in email\n- 📱 Highlight services\n- 🎯 Suggest upgrades")
            else:
                st.success("✅ **LOW PRIORITY**")
                st.markdown("- 😊 Maintain service\n- 📈 Cross-sell\n- 🎁 Loyalty rewards\n- 📞 Regular check-ins")
            
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)[:200]}")

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
**Model Performance:**
- Accuracy: 82.43%
- Precision: 70.38%
- ROC-AUC: 0.8476

**Top Risk Factors:**
1. Contract (Month-to-month)
2. Tenure (How long)
3. Internet Service Quality
""")