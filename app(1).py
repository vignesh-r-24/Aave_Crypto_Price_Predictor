
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
from datetime import date

pd.set_option('display.max_columns', None)

# Load dataset
df = pd.read_csv("coin_Aave.csv")


print(" Dataset Loaded Successfully\n")
print("🔸 Shape:", df.shape)
print("\n🔸 Column Names:\n", df.columns.tolist())
print("\n🔸 Data Types:\n")
print(df.dtypes)
print("\n🔸 Missing Values:\n")
print(df.isnull().sum())

for col in df.columns:
    if 'date' in col.lower():
        df[col] = pd.to_datetime(df[col], errors='coerce')

df.drop_duplicates(inplace=True)

print("\n Summary Statistics:\n")
print(df.describe())

numeric_cols = df.select_dtypes(include=np.number).columns
n_cols = len(numeric_cols)
n_rows = int(np.ceil(n_cols / 2))



print("\n EDA Completed Successfully ")

#model __build

model = joblib.load("crypto_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Aave Crypto Price Predictor", page_icon="💰", layout="centered")

st.title(" Aave Cryptocurrency Price Predictor")
st.write("Predict the next **Close Price** based on current market values.")

# User inputs
st.header(" Input Parameters")

open_price = st.number_input("Open Price", min_value=0.0, step=0.01)
high_price = st.number_input("High Price", min_value=0.0, step=0.01)
low_price = st.number_input("Low Price", min_value=0.0, step=0.01)
volume = st.number_input("Volume", min_value=0.0, step=0.01)

# Predict button
if st.button("Predict"):
    # Prepare input
    X_input = np.array([[open_price, high_price, low_price, volume]])
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)[0]

    st.success(f"📈 Predicted Close Price: **${prediction:,.2f}**")

# Optional: show recent data preview

fig, axes = plt.subplots(n_rows, 2, figsize=(12, n_rows * 3))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, bins=30, ax=axes[i], color='skyblue')
    # Shorten long titles automatically
    title_text = col if len(col) < 25 else col[:25] + "..."
    axes[i].set_title(f"Distribution of {title_text}", fontsize=11, pad=8)
    axes[i].set_xlabel(col, fontsize=8, labelpad=2)
    axes[i].set_ylabel("Frequency", fontsize=8, labelpad=2)

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout to prevent overlap
plt.tight_layout(pad=2.0)
fig.subplots_adjust(top=0.95, hspace=0.4, wspace=0.3)



if any(df.dtypes == 'datetime64[ns]'):
    date_col = df.select_dtypes(include='datetime64[ns]').columns[0]
    plt.figure(figsize=(10,5))
    for col in numeric_cols[:3]:  # show first 3 numeric features
        sns.lineplot(x=df[date_col], y=df[col], label=col)
    plt.title("Numeric Features Over Time", fontsize=13)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Values", fontsize=11)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)




# ==============================
# 💡 Crypto-Specific Plot (Close vs Volume)
# ==============================
if {'Close', 'Volume'}.issubset(df.columns):
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(df['Close'], color='tab:blue', label='Close Price')
    ax1.set_ylabel('Close Price', color='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(df['Volume'], color='tab:orange', label='Volume', alpha=0.7)
    ax2.set_ylabel('Volume', color='tab:orange')
    plt.title("Close Price vs Volume Trend", fontsize=13, pad=10)
    fig.tight_layout()
    st.pyplot(plt)
