# 🌱 GreenAlphaPredictor 📈

*A Deep Dive into Sustainable Finance using ESG for Price Forecasting*

---

## 🚀 Project Overview

**GreenAlphaPredictor** is not just another financial forecasting project.

It is a data-driven call to action:
> What if we could **forecast stock prices** not just based on market trends but on how responsibly a company behaves toward our 🌍 planet?

This project leverages **Environmental, Social, and Governance (ESG)** scores to forecast the future stock prices of **S&P 500** companies. But the true purpose is more profound than just prediction — it's about **highlighting the financial relevance of sustainability**.

---

## 🌍 Why ESG + Forecasting?

While Wall Street might chase profit, **GreenAlphaPredictor** chases **principles**.

📌 *The idea is simple yet bold:*  
> Can a company’s environmental stewardship, social responsibility, and governance ethics tell us something about its financial trajectory?

If yes, then sustainability isn't just **good ethics** — it's **smart economics**.

---

## 🛠️ Methodology

The project pipeline combines modern machine learning techniques with ESG analytics:

1. 📊 **Data Collection**
   - ESG data via [`yesg`](https://pypi.org/project/yesg/)
   - Price data from [`Yahoo Finance`](https://pypi.org/project/yfinance/)
   - Real-time ticker list from [Wikipedia - S&P 500 companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)

2. 🧠 **Modeling**
   - LSTM-based neural network architecture for time series forecasting
   - ESG features as input sequences, monthly price as the target
   - Global training across multiple companies for better generalization

3. 📈 **Training & Evaluation**
   - Optimized using `Adam`, learning rate scheduling, and gradient clipping
   - Evaluated via MSE, MAE, RMSE, MAPE and R² losses and plotted predictions vs actuals

4. 📦 **Clean Code & Modular Design**
   - Organized using `lib/` structure for scalability
   - Fully typed and documented classes: `DataLoader`, `TimeSeriesDataset`, `PriceForecastLSTM`

---

## 🧪 Experiments

✅ ESG scores were used to predict future stock price movements.  
✅ Training was performed across multiple S&P500 companies (19 used in default setup).  
✅ Model performance was visualized via loss curves and prediction plots.

> 🎯 The goal isn’t to outperform traditional models — but to demonstrate the **hidden signal** inside sustainability metrics.

---

## 📊 Key Outputs

- `plots/train_loss.png` — Training loss convergence  
- `plots/predicted_price.png` — Predicted vs Actual stock prices
- **Evaluation Results for S&P500 Companies:**
  - Mean Squared Error (MSE): 1.1886
  - Mean Absolute Error (MAE): 64.3742
  - Root Mean Squared Error (RMSE): 142.7451
  - Mean Absolute Percentage Error (MAPE): 79.52%
  - R² Score: 0.5616

---

## 🧾 Example Use

```bash
poetry install
poetry run python main.py
```

---

## 🤝 Let's Collaborate

If you are:

- 🔬 A researcher in **sustainable finance**  
- 🧠 A data scientist exploring **non-traditional time series inputs**  
- 🏦 A financial institution promoting **ethical AI**  
- 🌍 An environmental advocate interested in **data activism**  

then you’re invited to build on this work!  
Feel free to **fork** the repository, **open an issue**, or **reach out** to discuss ideas or collaborations.

---

### 🧑‍💻 Author

**Ahmet Çağatay Savaşlı**
[LinkedIn](https://www.linkedin.com/in/ahmet-cagatay-savasli-424a5a1b3/)

---

> _"In the age of algorithms, let’s teach our models not just to **think**, but to **care**."_ 🌱
