# 💰 Aave Cryptocurrency Price Predictor

A **Streamlit-based web app** that predicts the **next closing price** of the Aave cryptocurrency using a **machine learning model** trained on historical crypto market data.
This project combines **data analysis**, **model training**, and **interactive prediction visualization**.

---

## 📁 Project Structure

```
Aave-Crypto-Price-Predictor/
│
├── coin_Aave.csv               # Dataset used for model training & visualization
├── crypto_model.pkl            # Trained Random Forest model
├── scaler.pkl                  # StandardScaler used for feature normalization
├── app.py                      # Streamlit web app for predictions
├── model.py                    # Script to train and save ML model
└── README.md                   # Project documentation
```

---

## 🚀 Features

* 📊 **Exploratory Data Analysis (EDA)**: Automatically generates summary statistics and visualizations for numeric columns.
* 🤖 **Machine Learning Model**: Predicts the next **Close Price** using Random Forest regression.
* 🧠 **Feature Scaling**: Uses `StandardScaler` to normalize input features.
* 📈 **Interactive UI**: Users can input open, high, low, and volume values to predict price instantly.
* 📉 **Visualizations**:

  * Distributions of numeric columns
  * Time-series trends of key metrics
  * Relationship between **Close Price** and **Volume**

---

## 🧰 Technologies Used

| Category         | Libraries / Tools        |
| ---------------- | ------------------------ |
| Data Handling    | `pandas`, `numpy`        |
| Visualization    | `matplotlib`, `seaborn`  |
| Machine Learning | `scikit-learn`, `joblib` |
| Web App          | `streamlit`              |
| Date Handling    | `datetime`               |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Aave-Crypto-Price-Predictor.git
cd Aave-Crypto-Price-Predictor
```

### 2️⃣ Install Dependencies

Make sure you have **Python 3.8+** installed. Then run:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
pandas
numpy
matplotlib
seaborn
streamlit
scikit-learn
joblib
```

---

## 🧪 Model Training

If you want to train the model from scratch:

```bash
python model.py
```

This will:

* Load `coin_Aave.csv`
* Clean and process the data
* Train a `RandomForestRegressor`
* Save the model and scaler as `crypto_model.pkl` and `scaler.pkl`

---

## 🌐 Run the Web App

Once the model and scaler files are ready, launch the Streamlit app:

```bash
streamlit run app.py
```

Then open the displayed local URL (usually `http://localhost:8501`) in your browser.

---

## 🧭 How to Use

1. Enter values for:

   * **Open Price**
   * **High Price**
   * **Low Price**
   * **Volume**

2. Click **Predict** to see:

   * The **predicted closing price**
   * Several **data visualizations**

---

## 📊 Sample Output

**Example Prediction:**

```
📈 Predicted Close Price: $112.45
```

**Visualizations:**

* Feature distributions
* Time trends
* Close vs Volume relationship

---

## 🧾 License

This project is open-source under the **MIT License**.
Feel free to use, modify, and share!

---

## 👨‍💻 Author

**Your Name**
📧 [[your.email@example.com](mailto:your.email@example.com)]
🌐 [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

---


