# 🚗 AutoValuate AI — Quikr Car Price Predictor

An interactive Machine Learning web application built using **Streamlit**, **Scikit-Learn**, and **Plotly** to estimate used car resale market values based on the Quikr Cars dataset.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 🌟 Key Features

- **Accurate Price Valuation**: Machine Learning regression pipeline with One-Hot Categorical Encoding and Linear Regression.
- **Interactive UI**: Modern glassmorphic dark theme with real-time feedback.
- **Financial Estimators**: Built-in EMI monthly payment and total interest calculator.
- **Fuel Economy Benchmarks**: Automated ARAI mileage (km/L) estimator across brands and body styles.
- **Market Visualizations**: Interactive Plotly charts for manufacturer prices, depreciation curves, mileage impact, and fuel type distributions.

---

## 📁 Repository Structure

```
├── app.py                     # Streamlit web application frontend
├── utils.py                   # Helper functions, chart builders & estimators
├── Cleaned_Car_data.csv       # Cleaned dataset for vehicle metadata & lookup
├── LinearRegressionModel.pkl  # Trained Scikit-Learn regression pipeline
├── requirements.txt           # Python dependencies for deployment
└── README.md                  # Project documentation
```

---

## ⚙️ Local Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd "DATAANALYSIS WITH ML"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment on Streamlit Cloud

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"** or **"Create app"**.
4. Select your repository, set the branch to `main`, and enter `app.py` as the **Main file path**.
5. Click **"Deploy"**!
