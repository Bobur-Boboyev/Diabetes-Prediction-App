# Diabetes Prediction App

An interactive web application built with **Streamlit** that predicts the likelihood of diabetes based on medical information. This tool is designed for educational and demonstration purposes using machine learning models.

---

## About the Project

This project uses the **Pima Indians Diabetes Dataset** to build a predictive model that determines whether a person is likely to have diabetes based on input features such as:

- Pregnancies
- Glucose level
- Blood pressure
- Skin thickness
- Insulin level
- BMI
- Diabetes pedigree function
- Age

---

## How to Run the App

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Bobur-Boboyev/Diabetes-Prediction-App.git
   cd Diabetes-Prediction-App
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   streamlit run app/app.py
   ```

---

## Model Information

- **Model Used**: Logistic Regression
- **Scaler**: StandardScaler
- **Accuracy**: ~75% on test data
- **ROC AUC Score**: ~0.82

The model was trained on the cleaned and scaled dataset after replacing zero values in key medical fields with the median.

---

## Dataset

- **Source**: [Pima Indians Diabetes Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 768 samples with 8 input features and 1 output label.
