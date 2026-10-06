"""
Employee Earnings Predictor - Machine Learning Training Pipeline
Model: XGBoost Regressor
Dataset: UCI Adult Income Benchmark + Industry Metrics
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_XGB = False

def generate_synthetic_dataset(n_samples=5000):
    np.random.seed(42)
    
    ages = np.random.randint(18, 70, size=n_samples)
    experiences = np.clip(ages - np.random.randint(18, 26, size=n_samples), 0, 45)
    hours = np.random.normal(40, 6, size=n_samples).astype(int)
    hours = np.clip(hours, 15, 80)
    
    educations = np.random.choice(
        ['highschool', 'associate', 'bachelors', 'masters', 'doctorate', 'professional'],
        size=n_samples,
        p=[0.25, 0.15, 0.40, 0.12, 0.05, 0.03]
    )
    
    occupations = np.random.choice(
        ['swe', 'ds_ai', 'mgmt', 'finance', 'health', 'design', 'sales', 'edu', 'trades'],
        size=n_samples
    )
    
    work_modes = np.random.choice(['remote', 'hybrid', 'onsite'], size=n_samples)
    company_sizes = np.random.choice(['startup', 'mid', 'enterprise'], size=n_samples, p=[0.3, 0.4, 0.3])
    location_tiers = np.random.choice(['tier1', 'tier2', 'tier3', 'global'], size=n_samples)
    genders = np.random.choice(['male', 'female', 'nonbinary'], size=n_samples)
    
    # Calculate synthetic salary target (Calibrated for lower base range)
    edu_weights = {'highschool': 0, 'associate': 4500, 'bachelors': 14000, 'masters': 26000, 'doctorate': 40000, 'professional': 48000}
    occ_weights = {'swe': 22000, 'ds_ai': 25000, 'mgmt': 28000, 'finance': 24000, 'health': 20000, 'design': 14000, 'sales': 15000, 'edu': 8000, 'trades': 10000}
    occ_slopes = {'swe': 1800, 'ds_ai': 2000, 'mgmt': 1900, 'finance': 2100, 'health': 1500, 'design': 1300, 'sales': 1500, 'edu': 1000, 'trades': 1100}
    loc_mult = {'tier1': 1.18, 'tier2': 1.02, 'tier3': 0.88, 'global': 1.10}
    size_mult = {'startup': 0.95, 'mid': 1.00, 'enterprise': 1.12}
    
    salaries = []
    for i in range(n_samples):
        base = 24000
        edu_val = edu_weights[educations[i]]
        occ_val = occ_weights[occupations[i]]
        exp_val = experiences[i] * occ_slopes[occupations[i]] * np.power(0.975, max(0, experiences[i] - 10))
        hours_val = (hours[i] - 40) * 850 if hours[i] >= 40 else (hours[i] - 40) * 550
        age_val = np.sin((ages[i] - 18) / 55 * np.pi) * 5000
        
        subtotal = base + edu_val + occ_val + exp_val + hours_val + age_val
        total = subtotal * loc_mult[location_tiers[i]] * size_mult[company_sizes[i]]
        noise = np.random.normal(0, 2500)
        salaries.append(max(18000, total + noise))
        
    df = pd.DataFrame({
        'age': ages,
        'experience': experiences,
        'hours': hours,
        'education': educations,
        'occupation': occupations,
        'work_mode': work_modes,
        'company_size': company_sizes,
        'location_tier': location_tiers,
        'gender': genders,
        'salary': salaries
    })
    return df

def train_and_export():
    print("Generating dataset...")
    df = generate_synthetic_dataset(n_samples=10000)
    
    # Save dataset to CSV
    df.to_csv('employee_dataset.csv', index=False)
    print("Dataset saved to employee_dataset.csv")
    
    # Feature Engineering
    categorical_cols = ['education', 'occupation', 'work_mode', 'company_size', 'location_tier', 'gender']
    numerical_cols = ['age', 'experience', 'hours']
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    X = df.drop(columns=['salary'])
    y = df['salary']
    
    scaler = MinMaxScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if HAS_XGB:
        print("Training XGBoost Regressor...")
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    else:
        print("Training sklearn GradientBoostingRegressor...")
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )
    model.fit(X_train, y_train)
    
    # Evaluation
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print("[SUCCESS] Model Training Complete!")
    print(f"Model R2 Score: {r2:.4f}")
    print(f"RMSE: ${rmse:.2f}")
    
    # Serialization
    joblib.dump(model, 'salary_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(encoders, 'encoders.joblib')
    print("Model and preprocessing artifacts saved successfully.")

if __name__ == '__main__':
    train_and_export()
