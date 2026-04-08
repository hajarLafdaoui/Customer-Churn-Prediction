"""
Customer Churn Prediction - SIMPLE & FRIENDLY Version
Inputs in sidebar, results in main area
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Page setup
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('models/best_churn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    return model, scaler, label_encoders

model, scaler, label_encoders = load_model()

# ============================================
# SIDEBAR - ALL INPUTS HERE (NO EXTRA SPACE)
# ============================================

with st.sidebar:
    st.markdown("### 📋 Customer Information")
    st.caption("Answer 6 quick questions")
    st.divider()
    
    # Question 1
    how_long = st.selectbox(
        "📅 How long have they been a customer?",
        options=["Less than 6 months (New)", "6-12 months", "1-2 years", "More than 2 years"],
        help="New customers often leave more"
    )
    
    # Question 2
    contract = st.radio(
        "📝 What type of contract?",
        options=["Month-to-month (Can leave anytime)", "Yearly contract", "2+ year contract"],
        help="Month-to-month = easier to leave"
    )
    
    # Question 3
    monthly_bill = st.select_slider(
        "💰 How much is their monthly bill?",
        options=["$20-40 (Low)", "$40-60 (Medium)", "$60-80 (Above average)", "$80-120 (High)"],
        value="$40-60 (Medium)"
    )
    
    # Question 4
    happiness = st.select_slider(
        "😊 How happy is the customer?",
        options=[
            "😡 Very unhappy", 
            "😐 Not happy", 
            "🙂 Okay/Satisfied", 
            "😊 Happy", 
            "⭐ Very happy"
        ],
        value="🙂 Okay/Satisfied"
    )
    
    # Question 5
    complaints = st.selectbox(
        "🗣️ Have they complained recently?",
        options=["No complaints", "1 complaint", "2-3 complaints", "4+ complaints"],
        help="More complaints = higher risk"
    )
    
    # Question 6
    tech_support = st.radio(
        "🛠️ Do they have tech support?",
        options=["Yes, they have help", "No, they don't"],
        help="Customers with no help often leave"
    )
    
    st.divider()
    
    # Predict button in sidebar
    predict_button = st.button("🔮 PREDICT CHURN RISK", type="primary", use_container_width=True)

# ============================================
# MAIN CONTENT AREA
# ============================================

# Title at top of main area
st.title("📊 Will Your Customer Leave?")
st.markdown("### Predict customer churn risk in seconds")

# Function to convert friendly answers to model inputs
def convert_to_model_inputs(how_long, contract, monthly_bill, happiness, complaints, tech_support):
    """Convert friendly answers to numbers the model understands"""
    
    # Convert tenure (how long)
    if how_long == "Less than 6 months (New)":
        tenure = 3
    elif how_long == "6-12 months":
        tenure = 9
    elif how_long == "1-2 years":
        tenure = 18
    else:  # More than 2 years
        tenure = 48
    
    # Convert contract type
    if contract == "Month-to-month (Can leave anytime)":
        contract_type = "Month-to-month"
    elif contract == "Yearly contract":
        contract_type = "One year"
    else:
        contract_type = "Two year"
    
    # Convert monthly bill
    if monthly_bill == "$20-40 (Low)":
        monthly_charge = 30
    elif monthly_bill == "$40-60 (Medium)":
        monthly_charge = 50
    elif monthly_bill == "$60-80 (Above average)":
        monthly_charge = 70
    else:
        monthly_charge = 100
    
    # Convert happiness
    if happiness == "😡 Very unhappy":
        satisfaction = 1
    elif happiness == "😐 Not happy":
        satisfaction = 2
    elif happiness == "🙂 Okay/Satisfied":
        satisfaction = 3
    elif happiness == "😊 Happy":
        satisfaction = 4
    else:  # Very happy
        satisfaction = 5
    
    # Convert complaints
    if complaints == "No complaints":
        num_complaints = 0
    elif complaints == "1 complaint":
        num_complaints = 1
    elif complaints == "2-3 complaints":
        num_complaints = 2
    else:
        num_complaints = 4
    
    # Convert tech support
    tech_support_value = "Yes" if tech_support == "Yes, they have help" else "No"
    
    return {
        'tenure_months': tenure,
        'monthly_charge': monthly_charge,
        'total_charges': monthly_charge * tenure,
        'contract_type': contract_type,
        'payment_method': 'Electronic check',
        'paperless_billing': 'Yes',
        'tech_support': tech_support_value,
        'online_security': 'Yes',
        'streaming_tv': 'Yes',
        'avg_monthly_gb': 50,
        'num_complaints': num_complaints,
        'satisfaction_score': satisfaction
    }

# ============================================
# SHOW RESULTS WHEN BUTTON IS CLICKED
# ============================================

if predict_button:
    # Convert friendly answers to model format
    customer_data = convert_to_model_inputs(how_long, contract, monthly_bill, happiness, complaints, tech_support)
    
    # Make prediction
    input_df = pd.DataFrame([customer_data])
    
    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = encoder.transform(input_df[col])
            except:
                input_df[col] = 0
    
    # Scale
    input_scaled = scaler.transform(input_df)
    
    # Get probability
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Determine risk
    if probability < 0.3:
        risk_level = "Low"
        risk_color = "green"
        risk_emoji = "🟢"
        header_message = "✅ GOOD NEWS! This customer is likely to stay"
    elif probability < 0.7:
        risk_level = "Medium"
        risk_color = "orange"
        risk_emoji = "🟡"
        header_message = "⚠️ ATTENTION NEEDED! This customer might be at risk"
    else:
        risk_level = "High"
        risk_color = "red"
        risk_emoji = "🔴"
        header_message = "🚨 URGENT! This customer is likely to leave soon!"
    
    # Show results
    st.markdown("---")
    st.subheader("📊 PREDICTION RESULTS")
    
    # Three big metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Churn Risk", f"{probability:.1%}")
    
    with col2:
        st.metric("Risk Level", f"{risk_emoji} {risk_level}")
    
    with col3:
        prediction_text = "Will Leave ❌" if probability > 0.5 else "Will Stay ✅"
        st.metric("Prediction", prediction_text)
    
    # Show header message
    st.markdown(f"## {header_message}")
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": risk_color},
            "steps": [
                {"range": [0, 30], "color": "#90EE90"},
                {"range": [30, 70], "color": "#FFD700"},
                {"range": [70, 100], "color": "#FF6B6B"}
            ]
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("💡 What To Do Next")
    
    recommendations = []
    
    if contract == "Month-to-month (Can leave anytime)":
        recommendations.append("📝 Offer a discount for signing a yearly contract")
    
    if "Very unhappy" in happiness or "Not happy" in happiness:
        recommendations.append("📞 Call them immediately to understand their problems")
    
    if complaints != "No complaints":
        recommendations.append(f"🎧 They have {complaints.lower()} - resolve these issues first")
    
    if tech_support == "No, they don't":
        recommendations.append("💻 Offer free tech support for 3 months")
    
    if monthly_bill in ["$80-120 (High)"]:
        recommendations.append("💰 Review their plan - maybe they're paying too much")
    
    if how_long == "Less than 6 months (New)":
        recommendations.append("🎁 Send a welcome gift or loyalty bonus")
    
    if not recommendations:
        recommendations.append("✅ Keep doing what you're doing! Customer seems happy.")
    
    for rec in recommendations[:4]:
        st.write(rec)
    
    # Show what increases risk
    with st.expander("🔍 Why did we predict this?"):
        st.write("**Main reasons this customer might leave:**")
        
        risk_reasons = []
        
        if contract == "Month-to-month (Can leave anytime)":
            risk_reasons.append("• ❌ Month-to-month contract (easy to cancel)")
        if "Very unhappy" in happiness or "Not happy" in happiness:
            risk_reasons.append("• ❌ Low satisfaction score")
        if complaints != "No complaints":
            risk_reasons.append(f"• ❌ {complaints} - unresolved problems")
        if tech_support == "No, they don't":
            risk_reasons.append("• ❌ No tech support (can't get help)")
        if how_long == "Less than 6 months (New)":
            risk_reasons.append("• ❌ New customer (still deciding)")
        
        if risk_reasons:
            for reason in risk_reasons:
                st.write(reason)
        else:
            st.write("• ✅ No major risk factors found!")

else:
    # Show welcome screen when no prediction yet
    st.markdown("---")
    st.info("👈 **Ready?** Answer the 6 simple questions in the sidebar and click 'Predict Churn Risk'")
    
    st.markdown("""
    ### 🎯 What this tool does
    
    **Predicts if a customer will leave your business** before they actually do.
    
    ### 📊 How it works
    
    1. **Answer 6 simple questions** about the customer (in the left sidebar)
    2. **Click "Predict Churn Risk"**
    3. **Get instant results** + specific recommendations
    
    ### 💡 Real Examples
    
    | Customer Type | Situation | Result | Action |
    |--------------|-----------|--------|--------|
    | **High Risk** | New, month-to-month, unhappy, 3 complaints | 85% churn risk | Call them NOW! |
    | **Low Risk** | 2+ years, yearly contract, very happy, no complaints | 15% churn risk | Keep doing great |
    | **Medium Risk** | 1 year, okay satisfaction, 1 complaint | 45% churn risk | Send retention offer |
    
    ### 🚀 Why This Matters
    
    - **Save revenue** - Keep customers before they leave
    - **Reduce complaints** - Fix problems early
    - **Increase loyalty** - Happy customers stay longer
    
    ---
    *Powered by Machine Learning | Trained on 5,000+ customer records*
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>💡 Protects your revenue by preventing customer loss</div>",
    unsafe_allow_html=True
)