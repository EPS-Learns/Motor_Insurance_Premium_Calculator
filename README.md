# ⚡ Go Digit Spain | Motor Insurance Premium Pricing Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-yellow.svg)](https://www.python.org/downloads/)
[![GLM Actuarial Engine](https://img.shields.io/badge/actuarial-GLM%20Poisson%2FGamma-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive, production-ready **Actuarial Motor Insurance Tariff & Premium Calculator** built for **Go Digit Spain**. Powered by Generalized Linear Models (GLMs) trained on motor insurance claims experience datasets.

---

## 🌟 Key Features

- 🚘 **Multi-Peril Actuarial Pricing Engine**: Computes pure risk premiums ($\hat{\lambda} \times \hat{S}$) across 5 core motor coverages:
  1. ⚖️ **Third Party Liability** (Mandatory Legal Cover)
  2. 🚗 **Own Damage** (Property Collision & Impact)
  3. 🔑 **Theft Protection** (Total Loss & Stolen Parts)
  4. 🪟 **Windscreen & Glass** (Zero Deductible Glass Repair)
  5. 📜 **Legal Defense Cover** (Litigation & Legal Protection)
- 📊 **Commercial Underwriting Loadings**: Real-time breakdown of Net Risk Cost vs. Commercial Expenses and Target Profit Margins.
- 🎨 **Go Digit Spain Design System**: Modern light-yellow theme, high-contrast cards, and step-by-step risk profiling wizard.
- 📄 **Instant PDF Quote Download**: Generates official, multi-section policy quotation schedules (`fpdf2`).
- ☁️ **Cloud-Native Asset Management**: Automatic dynamic downloading of actuarial `.pkl` model files from GitHub Releases for seamless Streamlit Cloud deployment.

---

## 📐 Actuarial GLM Modeling Framework

The underlying rating engine utilizes a **Frequency-Severity actuarial framework**:

$$ \text{Gross Commercial Premium} = \frac{\sum_{\text{perils}} \left( \hat{\lambda}_i \times \hat{S}_i \right)}{1 - \text{Expense Ratio} - \text{Profit Margin}} $$

Where:
- **Claim Frequency ($\hat{\lambda}_i$)**: Modeled via **Poisson GLM** with log link function and exposure duration offsets $\log(\text{exposure})$.
- **Claim Severity ($\hat{S}_i$)**: Modeled via **Gamma GLM** with log link function on positive claim amounts.

| Coverage Peril | Frequency Model ($\hat{\lambda}$) | Severity Model ($\hat{S}$) | Exposure Term |
| :--- | :--- | :--- | :--- |
| **Third Party Liability** | Poisson GLM | Gamma GLM | `liability_exposure` |
| **Own Damage (Property)** | Poisson GLM | Gamma GLM | `total_exposure` |
| **Theft Protection** | Poisson GLM | Gamma GLM | `total_exposure` |
| **Windscreen & Glass** | Poisson GLM | Gamma GLM | `total_exposure` |
| **Legal Defense** | Poisson GLM | Gamma GLM | `total_exposure` |

---

## 🛠️ Repository Structure

```
Motor_Insurance_Premium_Calculator/
├── godigit_spain_app.py   # Primary Streamlit Application Script
├── glm.ipynb               # Actuarial GLM Model Training & Validation Notebook
├── requirements.txt        # Python Dependencies for Streamlit Cloud
├── .gitignore              # Ignores large .pkl binaries & cache files
└── README.md               # Documentation & Overview
```

---

## 🚀 Quick Start & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/EPS-Learns/Motor_Insurance_Premium_Calculator.git
cd Motor_Insurance_Premium_Calculator
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run godigit_spain_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Cloud

1. Fork or push this repository to GitHub.
2. Ensure model `.pkl` files are attached to your GitHub Release under tag `glm_model_pkl_files`.
3. Log in to [Streamlit Community Cloud](https://share.streamlit.io).
4. Click **New app** and select:
   - **Repository**: `EPS-Learns/Motor_Insurance_Premium_Calculator`
   - **Branch**: `main`
   - **Main file path**: `godigit_spain_app.py`
5. Click **Deploy**!

---

## 📜 License & Disclosures

Distributed under the **MIT License**. Tariffs are computer-generated using Generalized Linear Models parameterized on portfolio risk factors. Final policy issuance remains subject to underwriting guidelines and vehicle physical verification.
