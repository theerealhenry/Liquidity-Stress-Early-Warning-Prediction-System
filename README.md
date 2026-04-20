# 💧 Liquidity Stress Early Warning System
### 🚀 Production-Level Machine Learning Project | Financial Risk Analytics

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-yellow?logo=scikit-learn)
![Status](https://img.shields.io/badge/Project-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

This project builds a **machine learning-powered early warning system** to detect **customer liquidity stress within the next 30 days** using transactional mobile money data.

The goal is to move beyond static financial indicators and leverage **behavioral signals, temporal patterns, and transaction dynamics** to proactively identify at-risk customers.

---

## 🧠 Problem Statement

Traditional credit scoring models rely heavily on:
- Static demographic features  
- Historical credit performance  

However, **liquidity stress is dynamic** and often **behavior-driven**.

👉 This project answers:
> *Can we predict short-term liquidity stress using transaction behavior patterns?*

---

## 📊 Dataset Overview

The dataset is **high-dimensional, time-series structured**, containing:

### 🔹 Customer Profile Features
- `age`, `gender`, `region`
- `segment` (LVC / MVC / HVC)
- `earning_pattern` (Weekly / Monthly / Irregular)
- `smartphone` ownership
- `arpu` (Average Revenue Per User)

### 🔹 Behavioral Features (6-Month Window)
For each month (`m1` → `m6`):
- Paybill transactions  
- Merchant payments  
- Bank transfers  
- Mobile money sends  
- Money received  
- Deposits & withdrawals  

Each transaction type includes:
- Volume (frequency)
- Total value
- Highest transaction
- Unique counterparties

### 🔹 Liquidity Indicators
- `daily_avg_bal` (balance trends)
- `x_90_d_activity_rate` (engagement proxy)

### 🎯 Target Variable
- `liquidity_stress_next_30d` (binary classification)

---

## 🔍 Key Insights from Initial Analysis

### 1️⃣ Entity-Level Dataset (Not Time-Series Rows)
- Each row represents a **unique customer snapshot**
- Confirmed via Excel (`COUNTIF + FILTER`) → **no duplicate IDs**

📌 **Implication:**
> This is a **supervised classification problem**, not sequential modeling (yet).

---

### 2️⃣ Strong Temporal Structure (Hidden in Columns)
- Features span **6 months (m1–m6)**
- Encoded as **wide format time-series**

📌 **Opportunity:**
- Feature engineering:
  - Trends (increase/decrease)
  - Volatility
  - Behavioral shifts

---

### 3️⃣ Behavioral Segmentation Emerges Clearly

| Segment | Behavior Pattern |
|--------|----------------|
| **HVC** | Extremely high transaction volume & value |
| **MVC** | Moderate, structured usage |
| **LVC** | Sparse, low-frequency transactions |

📌 **Implication:**
> Segment-aware modeling will likely outperform a global model.

---

### 4️⃣ Transactional Diversity = Financial Complexity

Key behavioral dimensions:
- **Breadth:** number of merchants, agents, recipients
- **Depth:** transaction values and peaks
- **Consistency:** monthly continuity vs sparsity

📌 Users with:
- High diversity → financially active
- Low diversity → potentially constrained

---

### 5️⃣ Behavioral Imbalance & Sparsity

- Many zero values across months and channels
- Indicates:
  - Dormancy
  - Channel preference
  - Financial inactivity

📌 **Modeling implication:**
- Tree-based models will handle sparsity well
- Missing ≠ random → carries signal

---

### 6️⃣ Liquidity Stress is Behaviorally Driven

Early patterns observed:

⚠️ Potential stress signals:
- Declining balances (`daily_avg_bal ↓`)
- Reduced activity
- Increased withdrawals vs deposits
- Irregular inflows

💡 Strong candidates for predictive features:
- Balance volatility
- Net cash flow (inflows - outflows)
- Activity decay rates

---

### 7️⃣ High Feature Redundancy → Feature Engineering Required

For each transaction type, we have:
- Volume
- Total value
- Highest value
- Counterparty count

📌 These can be transformed into:
- Avg transaction value (`value / volume`)
- Transaction intensity
- Behavioral ratios

---

## 🧪 Initial Hypotheses

- Customers with **declining balances** are more likely to experience stress
- **Irregular earners** may show higher volatility → higher risk
- **Reduced transaction diversity** signals financial constraint
- **High withdrawals + low deposits** → liquidity drain

---

## 🏗️ Planned Pipeline

### Phase 1: Data Validation & Cleaning
- Missing values analysis
- Outlier detection
- Data consistency checks

### Phase 2: Feature Engineering
- Temporal trends (month-over-month changes)
- Aggregations (mean, std, min, max)
- Behavioral ratios
- Recency-weighted features

### Phase 3: Modeling
- Baseline: Logistic Regression
- Advanced:
  - Random Forest
  - XGBoost / LightGBM

### Phase 4: Evaluation
- ROC-AUC
- Precision-Recall (critical for imbalance)
- Business-aware metrics

### Phase 5: Deployment
- Streamlit dashboard
- Model API (optional)

---

## 🧰 Tech Stack

- **Python**
- **Pandas / NumPy**
- **Scikit-learn**
- **Matplotlib / Seaborn**
- **XGBoost / LightGBM**
- **Streamlit**

---

## 📁 Project Structure

**Initial Production-Level Project Structure**

```
liquidity-stress-early-warning/
│
├── .github/                    # CI/CD workflows 
│
├── configs/                   # Configuration files (YAML)
│   ├── config.yaml
│   └── model_params.yaml
│
├── data/
│   ├── raw/                   # Original immutable data
│   ├── interim/               # Cleaned but not final
│   └── processed/             # Final modeling data
│
├── notebooks/                 # experimental notebooks 
│   └── eda.ipynb
│
├── src/                       # Core source code 
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── validation.py
│   │   └── preprocessing.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   ├── pipelines/
│   │   ├── training_pipeline.py
│   │   └── inference_pipeline.py
│   │
│   ├── utils/
│   │   ├── logger.py
│   │   └── helpers.py
│
├── tests/                     # Unit tests 
│
├── artifacts/                 # Saved models, transformers
│
├── app/                       # deployment app
│
├── requirements.txt
├── setup.py                   # Makes project installable
├── .gitignore
└── README.md
```

---

## 🎯 Why This Project Stands Out

✔ Real-world financial problem  
✔ High-dimensional behavioral dataset  
✔ Time-aware feature engineering  
✔ Production-level structure  
✔ Business impact focus  

---

## 👨‍💻 Author

**Henry**  
Aspiring Machine Learning Engineer & Data Scientist  

---

## ⭐ Next Step

👉 Move into **deep EDA + feature engineering** to extract predictive signals from temporal behavioral patterns.

---