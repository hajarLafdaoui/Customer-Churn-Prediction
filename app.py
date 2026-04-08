"""
Customer Churn Prediction - Web Application
Interactive dashboard for predicting customer churn risk
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# ============================================
# PAGE CONFIGURATION (Must be first command)
# ============================================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD MODEL WITH CACHING
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        model = joblib.load('models/best_churn_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        return model, scaler, label_encoders
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run 'python src/train_model.py' first")
        st.stop()

# Load everything
model, scaler, label_encoders = load_model()

# ============================================
# HEADER SECTION
# ============================================
st.title("📊 Customer Churn Prediction System")
st.markdown("### Predict which customers are likely to leave your business")
st.markdown("---")

# ============================================
# SIDEBAR - INPUT FORM
# ============================================
with st.sidebar:
    st.header("🔧 Customer Information")
    st.markdown("Enter customer details below:")
    st.markdown("---")
    
    # Create form
    with st.form("prediction_form"):
        
        # CONTRACT INFORMATION
        st.subheader("📋 Contract Details")
        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
            help="How long has customer been with us?"
        )
        
        monthly_charge = st.slider(
            "Monthly Charge ($)",
            min_value=20.0,
            max_value=120.0,
            value=70.0,
            step=5.0,
            help="Monthly bill amount"
        )
        
        contract_type = st.selectbox(
            "Contract Type",
            options=["Month-to-month", "One year", "Two year"],
            help="Type of contract customer has"
        )
        
        payment_method = st.selectbox(
            "Payment Method",
            options=["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            help="How customer pays"
        )
        
        paperless_billing = st.selectbox(
            "Paperless Billing",
            options=["Yes", "No"],
            help="Is customer using paperless billing?"
        )
        
        st.markdown("---")
        
        # SERVICES
        st.subheader("🛠️ Services Subscribed")
        
        col1, col2 = st.columns(2)
        with col1:
            tech_support = st.selectbox(
                "Tech Support",
                options=["Yes", "No"],
                help="Does customer have tech support?"
            )
            online_security = st.selectbox(
                "Online Security",
                options=["Yes", "No"],
                help="Does customer have online security?"
            )
        
        with col2:
            streaming_tv = st.selectbox(
                "Streaming TV",
                options=["Yes", "No"],
                help="Does customer have streaming TV?"
            )
            avg_monthly_gb = st.slider(
                "Avg Monthly GB",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=5.0,
                help="Average data usage per month"
            )
        
        st.markdown("---")
        
        # CUSTOMER SATISFACTION
        st.subheader("😊 Customer Feedback")
        
        satisfaction_score = st.slider(
            "Satisfaction Score (1-5)",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = Very Unsatisfied, 5 = Very Satisfied"
        )
        
        num_complaints = st.number_input(
            "Number of Complaints",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="How many complaints in last month?"
        )
        
        st.markdown("---")
        
        # SUBMIT BUTTON
        submitted = st.form_submit_button(
            "🔮 Predict Churn Risk",
            type="primary",
            use_container_width=True
        )

# ============================================
# MAIN CONTENT - PREDICTION RESULTS
# ============================================

if submitted:
    # Create customer data dictionary
    customer_data = {
        'tenure_months': tenure,
        'monthly_charge': monthly_charge,
        'total_charges': monthly_charge * tenure,
        'contract_type': contract_type,
        'payment_method': payment_method,
        'paperless_billing': paperless_billing,
        'tech_support': tech_support,
        'online_security': online_security,
        'streaming_tv': streaming_tv,
        'avg_monthly_gb': avg_monthly_gb,
        'num_complaints': num_complaints,
        'satisfaction_score': satisfaction_score
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([customer_data])
    
    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = encoder.transform(input_df[col])
            except ValueError:
                # Handle unknown categories
                st.warning(f"Unknown value for {col}, using default")
                input_df[col] = 0
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Get prediction
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = "Will Churn" if probability > 0.5 else "Will Stay"
    
    # Determine risk level and color
    if probability < 0.3:
        risk_level = "Low"
        risk_color = "green"
        risk_emoji = "🟢"
    elif probability < 0.7:
        risk_level = "Medium"
        risk_color = "orange"
        risk_emoji = "🟡"
    else:
        risk_level = "High"
        risk_color = "red"
        risk_emoji = "🔴"
    
    # ========================================
    # DISPLAY RESULTS
    # ========================================
    
    st.markdown("## 📈 Prediction Results")
    st.markdown("---")
    
    # Three columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Churn Probability",
            value=f"{probability:.1%}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Risk Level",
            value=f"{risk_emoji} {risk_level} Risk",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Prediction",
            value=prediction,
            delta=None
        )
    
    # Gauge Chart
    st.markdown("---")
    st.subheader("📊 Risk Score Visualization")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={"text": "Churn Risk Score (%)"},
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": risk_color},
            "steps": [
                {"range": [0, 30], "color": "#90EE90"},
                {"range": [30, 70], "color": "#FFD700"},
                {"range": [70, 100], "color": "#FF6B6B"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    
    fig.update_layout(height=350, width=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================
    # RISK FACTORS ANALYSIS
    # ========================================
    st.markdown("---")
    st.subheader("⚠️ Risk Factors Detected")
    
    risk_factors = []
    recommendations = []
    
    # Check each risk factor
    if contract_type == "Month-to-month":
        risk_factors.append("🔴 Month-to-month contract (highest churn risk)")
        recommendations.append("📝 Offer discount for annual contract commitment")
    
    if satisfaction_score <= 2:
        risk_factors.append(f"🔴 Very low satisfaction score ({satisfaction_score}/5)")
        recommendations.append("📞 Schedule customer satisfaction call immediately")
    
    if num_complaints >= 2:
        risk_factors.append(f"🔴 Multiple complaints ({num_complaints} complaints)")
        recommendations.append("🎧 Escalate to customer support manager")
    
    if tech_support == "No":
        risk_factors.append("🔴 No tech support subscription")
        recommendations.append("💻 Offer free tech support trial for 3 months")
    
    if monthly_charge > 80:
        risk_factors.append(f"🔴 High monthly charges (${monthly_charge:.0f})")
        recommendations.append("💰 Review billing plan for potential savings")
    
    if tenure < 6:
        risk_factors.append("🔴 New customer (less than 6 months)")
        recommendations.append("🎁 Welcome bonus or loyalty points")
    
    if online_security == "No":
        risk_factors.append("🟡 No online security subscription")
        recommendations.append("🛡️ Offer security package at 50% discount")
    
    if avg_monthly_gb > 70:
        risk_factors.append("🟡 High data usage (risk of overage charges)")
        recommendations.append("📱 Recommend unlimited data plan")
    
    if payment_method == "Electronic check":
        risk_factors.append("🟡 Electronic check payment (higher default risk)")
        recommendations.append("💳 Offer discount for auto-pay setup")
    
    # Good factors (reduces churn)
    good_factors = []
    
    if contract_type == "Two year":
        good_factors.append("✅ Long-term contract (2 years)")
    
    if satisfaction_score >= 4:
        good_factors.append(f"✅ High satisfaction score ({satisfaction_score}/5)")
    
    if num_complaints == 0:
        good_factors.append("✅ No complaints filed")
    
    if tech_support == "Yes" and online_security == "Yes":
        good_factors.append("✅ Both tech support and security active")
    
    # Display risk factors
    if risk_factors:
        for factor in risk_factors:
            st.write(factor)
    else:
        st.write("✅ No major risk factors detected!")
    
    # Display good factors
    if good_factors:
        st.markdown("### 🌟 Positive Factors")
        for factor in good_factors:
            st.write(factor)
    
    # ========================================
    # RETENTION RECOMMENDATIONS
    # ========================================
    st.markdown("---")
    st.subheader("💡 Retention Recommendations")
    
    if probability > 0.7:
        st.error("### 🚨 URGENT ACTION REQUIRED")
        for rec in recommendations[:5]:  # Top 5 recommendations
            st.write(rec)
        st.write("🎯 **Priority:** High - Contact customer within 24 hours")
        
    elif probability > 0.3:
        st.warning("### ⚠️ Proactive Retention Recommended")
        for rec in recommendations[:3]:  # Top 3 recommendations
            st.write(rec)
        st.write("🎯 **Priority:** Medium - Contact customer within 1 week")
        
    else:
        st.success("### ✅ Customer Appears Satisfied")
        st.write("📌 Continue current engagement strategy")
        st.write("🎁 Consider loyalty program enrollment")
        st.write("📧 Send periodic satisfaction surveys")
    
    # ========================================
    # BUSINESS IMPACT
    # ========================================
    st.markdown("---")
    st.subheader("💰 Business Impact")
    
    # Calculate estimated impact
    monthly_revenue = monthly_charge
    yearly_revenue = monthly_revenue * 12
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Monthly Revenue at Risk",
            f"${monthly_revenue:.0f}",
            delta="if customer churns"
        )
    
    with col2:
        st.metric(
            "Yearly Revenue at Risk",
            f"${yearly_revenue:.0f}",
            delta="if customer churns"
        )
    
    # ROI of retention
    retention_cost = monthly_revenue * 0.2  # 20% discount offer
    savings_if_retained = monthly_revenue - retention_cost
    
    st.info(f"💡 **Insight:** Offering a 20% discount (${retention_cost:.0f}) could save ${savings_if_retained:.0f} monthly revenue")

else:
    # Show welcome screen when no prediction yet
    st.info("👈 **Get Started:** Enter customer information in the sidebar and click 'Predict Churn Risk'")
    
    # Show features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 What This Tool Does
        
        - **Predicts** which customers are likely to leave
        - **Identifies** key risk factors
        - **Provides** retention recommendations
        - **Calculates** business impact
        
        ### 📊 Key Churn Drivers
        
        - Month-to-month contracts
        - Low satisfaction scores
        - Multiple complaints
        - No tech support
        - High monthly charges
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 How to Use
        
        1. **Fill customer details** in the sidebar
        2. **Click "Predict Churn Risk"**
        3. **Review risk assessment**
        4. **Follow recommendations**
        
        ### 💡 Pro Tips
        
        - Try different contract types
        - See how satisfaction affects risk
        - Compare high vs low risk scenarios
        """)
    
    # Sample scenario buttons
    st.markdown("---")
    st.subheader("🎮 Try Sample Scenarios")
    
    col1, col2, col3 = st.columns(3)
    
    if col1.button("📱 High Risk Customer", use_container_width=True):
        st.session_state['sample'] = 'high'
    
    if col2.button("💼 Medium Risk Customer", use_container_width=True):
        st.session_state['sample'] = 'medium'
    
    if col3.button("⭐ Low Risk Customer", use_container_width=True):
        st.session_state['sample'] = 'low'

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Customer Churn Prediction System | Powered by Machine Learning</p>
        <p>© 2024 | For business use only</p>
    </div>
    """,
    unsafe_allow_html=True
)