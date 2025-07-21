import pandas as pd
import random

# Load the dataset
df = pd.read_csv("adult 3.csv")

# Check if income column exists
if 'income' not in df.columns:
    raise ValueError("Column 'income' not found.")

# Map categorical income to estimated salary ranges
def estimate_salary(income_label):
    if income_label.strip() == '>50K':
        return random.randint(51000, 120000)  # estimate for high earners
    else:
        return random.randint(20000, 49000)   # estimate for low earners

# Apply function
df['salary'] = df['income'].apply(estimate_salary)

# Drop old income column
df.drop(columns=['income'], inplace=True)

# Save to new CSV
df.to_csv("adult_salary.csv", index=False)
print("✅ Saved updated dataset as 'adult_salary.csv'")
