# 🏥 Diabetes Readmission Risk Predictor

An AI-powered healthcare analytics application that predicts the likelihood of a diabetic patient being readmitted to the hospital within 30 days of discharge.

Built using Machine Learning, Streamlit, and Google Gemini AI, this application helps healthcare professionals identify high-risk patients and make informed decisions for post-discharge care.

# Live App 
https://diabetes-readmission-predictor-v2.streamlit.app

## 🚀 Features

* Predicts 30-day hospital readmission risk
* Displays readmission probability score
* Shows model confidence metrics
* Interactive and user-friendly Streamlit dashboard
* AI-generated clinical recommendations using Google Gemini
* Real-time risk assessment based on patient information

## 🧠 Machine Learning Pipeline

### Data Preprocessing

* Missing value handling
* Feature engineering
* Categorical encoding
* Feature scaling using StandardScaler

### Models Evaluated

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

### Best Performing Model

**Random Forest Classifier**

| Metric   | Score  |
| -------- | ------ |
| Accuracy | 90.35% |
| F1 Score | 89.98% |
| ROC-AUC  | 95.19% |

## 🛠️ Tech Stack

* Python
* Streamlit
* Scikit-Learn
* Pandas
* NumPy
* Joblib
* Google Gemini API
* Python Dotenv

## 📂 Project Structure

```text
├── app.py
├── model.pkl
├── scaler.pkl
├── features.pkl
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## ⚙️ Installation

### Clone Repository

```bash
git clone git@github.com:pardhukatta/diabetes-readmission-predictor.git
cd diabetes-readmission-predictor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

### Run Application

```bash
streamlit run app.py
```

## 📈 Use Case

Hospitals can use this application to:

* Identify patients at high risk of readmission
* Improve discharge planning
* Enhance patient outcomes
* Reduce avoidable healthcare costs
* Support data-driven clinical decisions

## 🔒 Security

Sensitive credentials such as API keys are stored using environment variables and are excluded from version control through `.gitignore`.

## 📜 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Chandrakanth Katta**

Data Scientist | Machine Learning | Generative AI | NLP | RAG Applications

GitHub: https://github.com/pardhukatta

