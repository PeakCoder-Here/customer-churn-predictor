# 📊 Customer Churn Predictor

A machine learning web application that predicts whether a customer 
will churn based on their service usage patterns.

🚀 **Live App:** (https://customer-churn-predictor-hudmgakjobpf9en6i2bgr2.streamlit.app)

---

## 📌 Project Overview

Customer churn is one of the biggest challenges for subscription-based 
businesses. This project builds a predictive model to identify at-risk 
customers before they leave, enabling proactive retention strategies.

---

## 🎯 Results

| Metric | Score |
|--------|-------|
| Model | Random Forest |
| Accuracy | **79.25%** |
| Dataset Size | 7,032 records |
| Features Used | 19 |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data cleaning |
| Scikit-learn | ML model |
| Matplotlib & Seaborn | Visualizations |
| Streamlit | Web app deployment |
| Jupyter Notebook | Analysis environment |

---

## 📂 Project Structure

```
customer-churn-predictor/
├── app.py                  # Streamlit web app
├── Churn_Prediction.ipynb  # Full ML analysis notebook
├── model.pkl               # Trained Random Forest model
├── columns.pkl             # Feature column names
├── requirements.txt        # Dependencies
└── archive/
    └── churn.csv           # Dataset
```

---

## 🔍 Key Findings

- **TotalCharges** is the strongest predictor of churn
- **MonthlyCharges** and **Tenure** are the 2nd and 3rd most important
- Customers on **month-to-month contracts** churn at the highest rate
- New customers with **high monthly charges** are highest risk

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/PeakCoder-Here/customer-churn-predictor.git
cd customer-churn-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author

**Sudeep** — CS Student | Biratnagar, Nepal  
🌐 Live App: [customer-churn-predictor.streamlit.app](https://customer-churn-predictor-hudmgakjobpf9en6i2bgr2.streamlit.app)  
📁 GitHub: [@PeakCoder-Here](https://github.com/PeakCoder-Here)
