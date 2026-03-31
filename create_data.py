import pandas as pd
import numpy as np

# Set random seed for reproducible results
np.random.seed(42)

# Number of customers to generate
n_customers = 5000

# Generate customer data
print("Generating customer data...")

data = {
    'customer_id': range(1, n_customers + 1),
    'tenure_months': np.random.randint(1, 72, n_customers),
    'monthly_charge': np.random.uniform(20, 120, n_customers),
    'total_charges': np.random.uniform(100, 5000, n_customers),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers, p=[0.55, 0.25, 0.20]),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_customers),
    'paperless_billing': np.random.choice(['Yes', 'No'], n_customers),
    'tech_support': np.random.choice(['Yes', 'No'], n_customers, p=[0.45, 0.55]),
    'online_security': np.random.choice(['Yes', 'No'], n_customers, p=[0.50, 0.50]),
    'streaming_tv': np.random.choice(['Yes', 'No'], n_customers),
    'avg_monthly_gb': np.random.uniform(0, 100, n_customers),
    'num_complaints': np.random.poisson(0.3, n_customers),
    'satisfaction_score': np.random.randint(1, 6, n_customers),
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate churn probability based on realistic patterns
churn_probability = (
    (df['contract_type'] == 'Month-to-month') * 0.3 +
    (6 - df['satisfaction_score']) * 0.1 +
    df['num_complaints'] * 0.2 +
    (df['tech_support'] == 'No') * 0.15 +
    (df['monthly_charge'] > 80) * 0.1
)

# Cap probability between 0 and 1
churn_probability = np.clip(churn_probability, 0, 1)

# Assign churn based on probability
df['churn'] = np.random.binomial(1, churn_probability)

# Save to CSV file
df.to_csv('telecom_churn_data.csv', index=False)

# Print summary
print(f"✅ Dataset created successfully!")
print(f"📊 Total customers: {len(df)}")
print(f"📈 Churn rate: {df['churn'].mean()*100:.1f}%")
print(f"📁 File saved as: telecom_churn_data.csv")
print(f"\nFirst 5 rows of data:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
