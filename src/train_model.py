"""
Customer Churn Prediction - Model Training Script
This script trains and evaluates multiple ML models for churn prediction
"""

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Create necessary directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("=" * 60)
print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
print("=" * 60)

# ============================================
# STEP 1: LOAD DATASET
# ============================================
print("\n📂 Step 1: Loading dataset...")
df = pd.read_csv('data/telecom_churn_data.csv')
print(f"✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Churn rate: {df['churn'].mean()*100:.1f}%")

# ============================================
# STEP 2: PREPARE FEATURES AND TARGET
# ============================================
print("\n🔧 Step 2: Preparing features and target...")

# Separate features (X) and target (y)
X = df.drop(['churn', 'customer_id'], axis=1)
y = df['churn']

print(f"   Features: {X.shape[1]} columns")
print(f"   Target: {y.name}")

# ============================================
# STEP 3: ENCODE CATEGORICAL VARIABLES
# ============================================
print("\n📝 Step 3: Encoding categorical variables...")

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"   Found {len(categorical_cols)} categorical columns: {list(categorical_cols)}")

# Encode each categorical column
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"   ✓ Encoded: {col}")

# ============================================
# STEP 4: SPLIT DATA INTO TRAIN/TEST SETS
# ============================================
print("\n✂️ Step 4: Splitting data into train/test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintains class balance
)

print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Testing set: {X_test.shape[0]} samples")
print(f"   Train churn rate: {y_train.mean()*100:.1f}%")
print(f"   Test churn rate: {y_test.mean()*100:.1f}%")

# ============================================
# STEP 5: SCALE NUMERICAL FEATURES
# ============================================
print("\n📊 Step 5: Scaling numerical features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Features scaled successfully!")

# ============================================
# STEP 6: TRAIN MULTIPLE MODELS
# ============================================
print("\n🤖 Step 6: Training models...")
print("-" * 40)

# Dictionary to store models and their results
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

results = {}

for name, model in models.items():
    print(f"\n   Training {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred
    }
    
    print(f"   ✓ Accuracy: {accuracy:.4f}")
    print(f"   ✓ Precision: {precision:.4f}")
    print(f"   ✓ Recall: {recall:.4f}")
    print(f"   ✓ F1-Score: {f1:.4f}")

# ============================================
# STEP 7: SELECT BEST MODEL
# ============================================
print("\n🏆 Step 7: Selecting best model...")
print("-" * 40)

best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']
best_f1 = results[best_model_name]['f1']

print(f"   Best model: {best_model_name}")
print(f"   Best F1-Score: {best_f1:.4f}")

# Display all models comparison
print("\n   Model Comparison:")
print("   " + "-" * 50)
for name, metrics in results.items():
    print(f"   {name:20} | F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.4f}")

# ============================================
# STEP 8: FEATURE IMPORTANCE (For Tree-based Models)
# ============================================
print("\n📊 Step 8: Analyzing feature importance...")

if best_model_name in ['Random Forest', 'XGBoost']:
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n   Top 10 Most Important Features:")
    print("   " + "-" * 40)
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:20} : {row['importance']:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top 10 Features Driving Customer Churn\n({best_model_name})', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n   ✅ Feature importance chart saved: outputs/feature_importance.png")
    plt.show()
else:
    print(f"   Feature importance not available for {best_model_name}")

# ============================================
# STEP 9: CONFUSION MATRIX
# ============================================
print("\n📈 Step 9: Creating confusion matrix...")

y_pred_best = results[best_model_name]['predictions']
cm = confusion_matrix(y_test, y_pred_best)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Stay', 'Churn'],
            yticklabels=['Stay', 'Churn'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()

# Save the plot
plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Confusion matrix saved: outputs/confusion_matrix.png")
plt.show()

# Display classification report
print("\n   Detailed Classification Report:")
print("   " + "-" * 40)
print(classification_report(y_test, y_pred_best, target_names=['Stay', 'Churn']))

# ============================================
# STEP 10: SAVE MODELS AND PREPROCESSING OBJECTS
# ============================================
print("\n💾 Step 10: Saving models and preprocessing objects...")
print("-" * 40)

# Save best model
joblib.dump(best_model, 'models/best_churn_model.pkl')
print(f"   ✅ Best model saved: models/best_churn_model.pkl")

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print(f"   ✅ Scaler saved: models/scaler.pkl")

# Save label encoders
joblib.dump(label_encoders, 'models/label_encoders.pkl')
print(f"   ✅ Label encoders saved: models/label_encoders.pkl")

# Save model metrics for reference
metrics_df = pd.DataFrame(results).T
metrics_df.to_csv('outputs/model_metrics.csv')
print(f"   ✅ Model metrics saved: outputs/model_metrics.csv")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\n📊 Best Model: {best_model_name}")
print(f"📈 F1-Score: {best_f1:.4f}")
print(f"🎯 Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"\n📁 Files saved:")
print(f"   • models/best_churn_model.pkl")
print(f"   • models/scaler.pkl")
print(f"   • models/label_encoders.pkl")
print(f"   • outputs/feature_importance.png")
print(f"   • outputs/confusion_matrix.png")
print(f"   • outputs/model_metrics.csv")
print("\n🚀 Ready to make predictions!")

# Optional: Test prediction on sample customer
print("\n" + "=" * 60)
print("🔍 Quick Test Prediction")
print("=" * 60)

# Create a sample customer
sample_customer = pd.DataFrame([{
    'tenure_months': 12,
    'monthly_charge': 85.5,
    'total_charges': 1026,
    'contract_type': 'Month-to-month',
    'payment_method': 'Electronic check',
    'paperless_billing': 'Yes',
    'tech_support': 'No',
    'online_security': 'No',
    'streaming_tv': 'Yes',
    'avg_monthly_gb': 45.2,
    'num_complaints': 2,
    'satisfaction_score': 2
}])

# Encode sample
for col, encoder in label_encoders.items():
    if col in sample_customer.columns:
        sample_customer[col] = encoder.transform(sample_customer[col])

# Scale sample
sample_scaled = scaler.transform(sample_customer)

# Predict
prob = best_model.predict_proba(sample_scaled)[0][1]
pred = "Churn" if prob > 0.5 else "Stay"

print(f"\n   Sample Customer:")
print(f"   • Contract: Month-to-month")
print(f"   • Satisfaction: 2/5")
print(f"   • Complaints: 2")
print(f"   • Tech Support: No")
print(f"\n   Prediction: {pred}")
print(f"   Churn Probability: {prob:.2%}")

print("\n" + "=" * 60)