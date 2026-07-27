# 🛡️ Advanced Credit Card Fraud Detection System 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Framework](https://img.shields.io/badge/UI-Streamlit-red)

## 📌 Project Overview
This is a B.Tech 3rd-year Machine Learning project focused on detecting fraudulent credit card transactions. 

Real-world credit card data is highly imbalanced (usually 99.8% valid vs. 0.2% fraud). A standard machine learning model will fail to detect fraud in such scenarios. This project tackles this challenge using **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the data and utilizes an interactive web dashboard for real-time transaction monitoring and analysis.

## ✨ Standout Features
Unlike standard ML scripts, this project is packaged as a complete software application with the following advanced features:

* **🧠 Explainable AI (XAI):** The dashboard doesn't just predict "Fraud" or "Valid"—it generates a dynamic chart showing *exactly which features* (e.g., V14, Transaction Amount) influenced the model's decision.
* **⚖️ Live Model Comparison:** Switch between **Random Forest** (High Accuracy, Non-linear) and **Logistic Regression** (High Speed, Linear) in real-time to compare their risk scores.
* **🤖 Auto-Fill Transaction Simulator:** No need to guess PCA feature values! One-click buttons instantly load real valid or fraudulent transactions from the dataset into the UI for testing.
* **📜 Transaction History Log:** Maintains a session-based "mini-statement" of all analyzed transactions with their respective risk scores and final decisions.

## 🛠️ Tech Stack Used
* **Programming Language:** Python
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, Imbalanced-learn (SMOTE)
* **Web Framework:** Streamlit

## 🚀 How to Run the Project Locally

**⚠️ Important Note:** The dataset is extremely large (150MB) and cannot be hosted on GitHub. You must download it manually before running the application.

### Step 1: Clone the Repository
```bash
git clone [https://github.com/Yogeshverma25/Credit-Card-Fraud-Detection.git](https://github.com/Yogeshverma25/Credit-Card-Fraud-Detection.git)
cd Credit-Card-Fraud-Detection

### Step 2: Download the Dataset
Go to the official Kaggle Dataset: Credit Card Fraud Detection (MLG-ULB)

Download and extract the ZIP file.

Place the creditcard.csv file directly into your project's main folder (in the same folder as app.py).

### Step 3: Install Dependencies
Open your terminal and run the following command to install required libraries:

Bash
pip install streamlit pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

### Step 4: Run the Dashboard
Bash
streamlit run app.py
📊 The Machine Learning Pipeline
Data Ingestion: Loaded the highly imbalanced dataset (284,315 valid vs. 492 fraud).

Exploratory Data Analysis (EDA): Visualized the class imbalance and plotted a correlation heatmap to understand the PCA-transformed features.

Data Balancing: Applied SMOTE on the training data to synthetically generate fraud cases, allowing the model to learn the fraud patterns accurately.

Model Training: Trained and serialized (.pkl) two distinct models:

Random Forest Classifier: (Primary Engine) Uses an ensemble of decision trees for high accuracy.

Logistic Regression: (Secondary Engine) Used as a baseline linear model for comparison.

👨‍💻 Developed By
Yogesh - B.Tech 3rd Year
