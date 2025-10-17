# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
df = pd.read_csv("coin_Aave.csv")

# Clean and prepare
df = df.dropna()
for col in df.columns:
    if 'date' in col.lower():
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Select features
features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

df = df.dropna(subset=features + [target])
X = df[features]
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "crypto_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model trained and saved successfully!")
