import pandas as pd
import joblib

# Load your original dataset
df = pd.read_csv("adult 3.csv")  # or whatever file you used for training

# Drop the target/output column
X = df.drop("income", axis=1)  # change "income" if your target column is named differently

# Get feature names
feature_names = list(X.columns)

# Save feature names to a file
joblib.dump(feature_names, "feature_names.pkl")

print("✅ Feature names saved:", feature_names)
