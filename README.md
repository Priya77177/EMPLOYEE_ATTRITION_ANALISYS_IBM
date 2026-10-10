# 🏢 Employee Attrition Analysis

## 🎯 Project Overview
This project focuses on HR Data Analytics to understand and predict employee attrition (turnover). Using **Exploratory Data Analysis (EDA)** and **Machine Learning** Classification models, it identifies the key factors that drive employees to leave the company (e.g., Job Satisfaction, Monthly Income, Distance from Home). 

Built as part of the **IBM SkillsBuild Data Analytics Internship**.

---

## 📁 Project Structure
```
Employee-Attrition-Analysis/
├── employee_attrition_analysis.py    # Main Streamlit application code
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── Employee_Attrition_Analysis_Report.md                 # Detailed project report
```

*(Note: The dataset is synthetically generated within the app to ensure out-of-the-box execution without external CSV dependencies, matching standard IBM HR Attrition dataset schemas.)*

---

## 🛠️ Technologies Used
| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core programming language |
| **Pandas & NumPy** | Data manipulation & numerical computing |
| **Plotly & Seaborn** | Interactive and statistical visualizations |
| **Scikit-learn** | Machine Learning Classification models |
| **Streamlit** | Web application framework |

---

## 🤖 Machine Learning Models
The project implements **4 Classification models**:
1. **Logistic Regression** — Baseline model
2. **Random Forest Classifier** — Ensemble tree-based model (Best for feature importance)
3. **Gradient Boosting Classifier** — Sequential boosting model
4. **Support Vector Machine (SVC)** — High-dimensional classifier

**Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

---

## 🚀 How to Run

### 1. Clone this repository
```bash
git clone https://github.com/Priya77177/EMPLOYEE_ATTRITION_ANALISYS_IBM.git
cd EMPLOYEE_ATTRITION_ANALISYS_IBM
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run employee_attrition_analysis.py
```

---

## 👤 Author
- **Name:** Priya Ranjan Pandey
- **Program:** IBM SkillsBuild Data Analytics Internship
- **Date:** October 2026

---
## 📄 License
Created for educational purposes as part of the IBM SkillsBuild Internship Program.
