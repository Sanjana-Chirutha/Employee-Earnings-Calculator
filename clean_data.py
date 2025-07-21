# ✅ clean_data.py (updated for salary regression)
import pandas as pd

# Load your updated dataset
df = pd.read_csv("adult_salary.csv")

# Drop rows with missing salary
df.dropna(subset=['salary'], inplace=True)

# Keep only required columns
features = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'gender', 'hours-per-week']
X = df[features]
y = df['salary']

# Save cleaned data for training and app
df[features + ['salary']].to_csv("clean_data.csv", index=False)

print("✅ clean_data.csv for regression generated successfully")
