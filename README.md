# 🧑‍💼 Employee Attrition Predictor

An AI-powered web application that predicts the likelihood of an employee leaving the company based on various HR metrics. This tool helps HR departments retain talent by proactively identifying high-risk employees.

---

## 🚀 Features

- Predict employee attrition using Machine Learning
- Clean and interactive **Streamlit UI**
- Customized prediction threshold for better **recall**
- Visual feedback and suggestions based on prediction
- Easy-to-use sliders and dropdowns for input
- Real-time result display with recommendations

---

## 🛠️ Technologies Used

| Component         | Tech                             |
|------------------|----------------------------------|
| Machine Learning | RandomForestClassifier (with GridSearchCV) |
| UI / Frontend    | Streamlit + HTML/CSS             |
| Data Processing  | Pandas, NumPy, SMOTE              |
| Model Deployment | Joblib                           |
| Graphs           | Matplotlib / Seaborn (optional)  |

---

## 🧑‍💻 How it Works

1. Load trained model (`final_model.joblib`)
2. Accept employee inputs (satisfaction, evaluation, department, etc.)
3. Preprocess and one-hot encode values
4. Predict attrition risk using ML model
5. Display probability and recommended HR actions

---


## Install dependencies
  - pip install -r requirements.txt

## Run the Streamlit app
  - streamlit run app.py


