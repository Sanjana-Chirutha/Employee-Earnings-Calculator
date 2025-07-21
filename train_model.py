# ✅ train_model.py for salary regression
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the cleaned dataset
df = pd.read_csv("clean_data.csv")

X = df.drop("salary", axis=1)
y = df["salary"]

# Preprocessing
categorical = X.select_dtypes(include="object").columns
numerical = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="mean"), numerical),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor())
])

# Train and save model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "salary_model.pkl")
print("✅ salary_model.pkl (regression) saved")
