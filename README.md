# 🎯 Holiday Package Purchase Prediction

A machine learning project that predicts whether a customer is likely to purchase a holiday package based on their demographics, engagement behavior, and interaction history. Deployed using **Streamlit** with a clean user interface for interactive predictions.

---

## 📁 Dataset

The dataset was sourced from Kaggle:  
**[Holiday Package Purchase Prediction](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction)**

It includes features such as:
- Age, Gender, Marital Status
- Number of Followups, Duration of Pitch
- Income, Designation, City Tier
- Satisfaction score, Product pitched

---

## 🧠 Model

We used a **Random Forest Classifier** and applied:
- One-hot encoding for categorical features (via `pd.get_dummies`)
- Class imbalance handling with `class_weight='balanced'`
- Hyperparameter tuning using **RandomizedSearchCV**

After training, the best estimator was exported using `joblib`.

---

## 🔍 Evaluation Metrics

- **Accuracy**
- **Confusion Matrix**
- **Classification Report**
- **Train/Test split**
- Achieved ~90% accuracy with good recall on the minority class after rebalancing.

---

## 🚀 Streamlit Deployment

An interactive UI was built using **Streamlit** where users can:
- Input customer details manually
- Get a binary prediction: Will purchase / Won’t purchase
- View model confidence (predicted probability)

### 🔧 Features included:
- Real-time prediction
- Manual feature input (Age, Gender, Income, Followups, etc.)
- Backend reindexing to match training feature space
- Automatic handling of missing dummy variables

---

## 🧪 Sample Input

| Field                | Value            |
|---------------------|------------------|
| Age                 | 28               |
| Gender              | Female           |
| Annual Income       | 10.0 (in Lakhs)  |
| Product Pitched     | Super Deluxe     |
| Pitch Satisfaction  | 5                |
| Designation         | Manager          |

---

## 🗂 Project Structure
holiday_package_prediction/
├── app.py # Streamlit web app
├── rf_model.pkl # Trained Random Forest model
├── model_features.json # Saved feature columns for prediction alignment
├── requirements.txt # Python dependencies
└── README.md # Project documentation




---

## 📦 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/holiday-package-prediction.git
   cd holiday-package-prediction

pip install -r requirements.txt

streamlit run app.py

✅ Requirements
Python 3.8+

pandas

scikit-learn

joblib

streamlit

📌 TODOs
Add support for CSV batch predictions

Display top feature importances in the app

Add user feedback logging and analytics

✨ Acknowledgements
Dataset by Susanta Sahu via Kaggle

Inspired by customer targeting use cases in marketing and sales
