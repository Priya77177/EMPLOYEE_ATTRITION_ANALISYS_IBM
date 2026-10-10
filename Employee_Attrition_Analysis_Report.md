# ============================================================================
#              EMPLOYEE ATTRITION ANALYSIS — PROJECT REPORT
#              IBM SkillsBuild Data Analytics Internship
# ============================================================================
#
# Author  : Priya Ranjan Pandey
# Date    : October 2026
# Program : IBM SkillsBuild Data Analytics Internship
#
# NOTE: This report is in Markdown format. To convert to PDF:
#       Option 1: Open in VS Code → Ctrl+Shift+P → "Markdown: Export to PDF"
#       Option 2: Paste into Google Docs and export as PDF
#
# ============================================================================


# PROJECT REPORT: Employee Attrition Analysis

---

## 1. INTRODUCTION

### 1.1 Background
Employee attrition (turnover) is a major challenge for modern organizations. High attrition rates lead to increased recruitment costs, loss of institutional knowledge, and decreased overall productivity. By leveraging Data Analytics and Machine Learning, HR departments can proactively identify employees at risk of leaving and understand the key drivers of attrition.

### 1.2 Objective
- To perform Exploratory Data Analysis (EDA) on HR data.
- To identify correlations between employee features (e.g., Job Satisfaction, Monthly Income) and attrition.
- To build Machine Learning classification models to predict whether an employee will leave.
- To deploy an interactive dashboard for HR managers.

---

## 2. DATASET DESCRIPTION

The project uses a synthetic HR dataset modeled after the standard IBM HR Attrition dataset, generated internally to ensure reproducibility.

**Key Features:**
1. **Age**: Employee's age
2. **Department**: Sales, R&D, HR, or IT
3. **Distance_From_Home**: Commute distance
4. **Education**: Education level (High School to PhD)
5. **Job_Satisfaction**: Rating from 1 (Low) to 4 (High)
6. **Monthly_Income**: Salary in USD
7. **Years_At_Company**: Tenure at the organization
8. **Attrition (Target)**: "Yes" (Left) or "No" (Stayed)

---

## 3. METHODOLOGY

### 3.1 Workflow
`Raw Data Generation → Preprocessing & Encoding → EDA → ML Model Training → Evaluation → Web Dashboard`

### 3.2 Machine Learning Models
This is a binary classification problem. Four models were evaluated:
1. **Logistic Regression** (Baseline)
2. **Random Forest Classifier** (Good for feature importance)
3. **Gradient Boosting Classifier** (Usually yields highest accuracy)
4. **Support Vector Machine** (Good for complex margins)

**Data Preprocessing:**
- Categorical variables encoded using `LabelEncoder`.
- Numerical variables scaled using `StandardScaler`.
- Data split: 80% Training, 20% Testing.

---

## 4. KEY FINDINGS (EDA)

1. **Job Satisfaction:** Employees with Low Job Satisfaction (Levels 1 & 2) show a significantly higher rate of attrition compared to those with High Satisfaction (Levels 3 & 4).
2. **Monthly Income:** Attrition is heavily skewed towards employees in the lower income brackets.
3. **Departmental Trends:** Sales and IT departments tend to have slightly higher turnover compared to R&D.
4. **Distance From Home:** Employees commuting longer distances show a higher propensity to leave.

---

## 5. MODEL EVALUATION

- **Best Performing Model:** Random Forest Classifier & Gradient Boosting Classifier typically achieve ~85%+ accuracy.
- **Key Metrics:** Because Attrition is often an imbalanced class (more "No" than "Yes"), **Precision, Recall, and F1-Score** are prioritized over raw accuracy.
- **Feature Importance:** Machine learning models confirm that `Monthly_Income`, `Job_Satisfaction`, and `Age` are the strongest predictors of attrition.

---

## 6. RECOMMENDATIONS FOR HR

Based on the data-driven insights:
1. **Targeted Retention Budgets:** Focus salary increments and bonuses on lower-income bands where attrition risk is highest.
2. **Improve Workplace Culture:** Address the root causes of low Job Satisfaction scores through surveys and management training.
3. **Remote Work Policies:** Offer flexible or remote working options to employees who have long commute distances.

---

## 7. CONCLUSION

The project successfully demonstrates how HR data can be transformed into actionable business intelligence. The deployed Streamlit application provides a user-friendly interface for HR professionals to explore data trends and utilize predictive machine learning models without needing coding expertise.

---
**Report prepared by:** Priya Ranjan Pandey  
**Date:** October 2026  
**Program:** IBM SkillsBuild Data Analytics Internship
