"""
Customer Churn Prediction - Prediction Module
Simple class to predict if a customer will leave
"""

import pandas as pd
import joblib
import os

class ChurnPredictor:
    """
    Simple churn prediction class
    Loads model and predicts customer churn risk
    """
    
    def __init__(self):
        """Load all saved models and tools"""
        print("Loading churn prediction model...")
        
        # Load the trained model
        self.model = joblib.load('models/best_churn_model.pkl')
        
        # Load the scaler (for normalizing numbers)
        self.scaler = joblib.load('models/scaler.pkl')
        
        # Load the label encoders (for converting text to numbers)
        self.label_encoders = joblib.load('models/label_encoders.pkl')
        
        print("✅ Model loaded successfully!")
    
    def predict(self, customer):
        """
        Predict churn risk for one customer
        
        Args:
            customer: Dictionary with customer information
            
        Returns:
            Dictionary with probability, risk level, and prediction
        """
        
        # Step 1: Convert dictionary to DataFrame
        df = pd.DataFrame([customer])
        
        # Step 2: Encode categorical fields (convert text to numbers)
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])
        
        # Step 3: Scale numerical features (normalize numbers)
        df_scaled = self.scaler.transform(df)
        
        # Step 4: Get churn probability from model
        probability = self.model.predict_proba(df_scaled)[0][1]
        
        # Step 5: Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Step 6: Make final prediction
        prediction = "Will Churn" if probability > 0.5 else "Will Stay"
        
        # Return results
        return {
            'probability': round(probability * 100, 1),  # as percentage
            'risk_level': risk_level,
            'prediction': prediction
        }


# ============================================
# QUICK TEST - Run this file directly
# ============================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("TESTING CHURN PREDICTOR")
    print("="*50)
    
    # Create predictor
    predictor = ChurnPredictor()
    
    # Test customer 1: HIGH RISK (should churn)
    high_risk_customer = {
        'tenure_months': 3,
        'monthly_charge': 95.0,
        'total_charges': 285,
        'contract_type': 'Month-to-month',
        'payment_method': 'Electronic check',
        'paperless_billing': 'Yes',
        'tech_support': 'No',
        'online_security': 'No',
        'streaming_tv': 'Yes',
        'avg_monthly_gb': 80.5,
        'num_complaints': 3,
        'satisfaction_score': 1
    }
    
    # Test customer 2: LOW RISK (should stay)
    low_risk_customer = {
        'tenure_months': 48,
        'monthly_charge': 55.0,
        'total_charges': 2640,
        'contract_type': 'Two year',
        'payment_method': 'Credit card',
        'paperless_billing': 'No',
        'tech_support': 'Yes',
        'online_security': 'Yes',
        'streaming_tv': 'No',
        'avg_monthly_gb': 25.0,
        'num_complaints': 0,
        'satisfaction_score': 5
    }
    
    # Test customer 3: MEDIUM RISK
    medium_risk_customer = {
        'tenure_months': 15,
        'monthly_charge': 75.0,
        'total_charges': 1125,
        'contract_type': 'One year',
        'payment_method': 'Bank transfer',
        'paperless_billing': 'Yes',
        'tech_support': 'No',
        'online_security': 'Yes',
        'streaming_tv': 'Yes',
        'avg_monthly_gb': 50.0,
        'num_complaints': 1,
        'satisfaction_score': 3
    }
    
    # Test all customers
    print("\n📊 TEST RESULTS:")
    print("-" * 50)
    
    result = predictor.predict(high_risk_customer)
    print(f"\n🔴 HIGH RISK CUSTOMER:")
    print(f"   Churn Probability: {result['probability']}%")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Prediction: {result['prediction']}")
    
    result = predictor.predict(medium_risk_customer)
    print(f"\n🟡 MEDIUM RISK CUSTOMER:")
    print(f"   Churn Probability: {result['probability']}%")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Prediction: {result['prediction']}")
    
    result = predictor.predict(low_risk_customer)
    print(f"\n🟢 LOW RISK CUSTOMER:")
    print(f"   Churn Probability: {result['probability']}%")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Prediction: {result['prediction']}")
    
    print("\n" + "="*50)
    print("✅ Predictor test complete!")
    print("="*50)