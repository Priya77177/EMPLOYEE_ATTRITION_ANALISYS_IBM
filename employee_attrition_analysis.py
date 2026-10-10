# ============================================================================
# Employee Attrition Analysis — IBM SkillsBuild Internship Project
# ============================================================================
# Author  : Priya Ranjan Pandey
# Date    : October 2026
# Dataset : Synthetic HR Attrition Data (Inspired by IBM HR Dataset)
# ============================================================================
# This project performs exploratory data analysis and predicts employee
# attrition using Python, Pandas, Scikit-learn, and Streamlit.
# ============================================================================

# ── 1. Imports ──────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import warnings
import os

warnings.filterwarnings("ignore")

# ── Lazy ML imports (to handle AppLocker / DLL restrictions) ────────────────
_sklearn_available = False
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )
    _sklearn_available = True
except (ImportError, OSError):
    class LabelEncoder:
        def __init__(self):
            self._classes = {}
        def fit_transform(self, data):
            unique = sorted(set(data))
            self._classes = {v: i for i, v in enumerate(unique)}
            return np.array([self._classes[v] for v in data])
        def transform(self, data):
            return np.array([self._classes.get(v, 0) for v in data])


# ── 2. Streamlit Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Analysis",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 3. Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    .main { background-color: #f8f9fa; color: #212529; font-family: 'Roboto', sans-serif; }
    .stApp { background-color: #f8f9fa; }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .main-header h1 { font-size: 2.5rem; margin: 0; font-weight: 700; color: white; }
    .main-header p { font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem; color: white; }

    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 2.2rem; font-weight: 700; color: #2c5364;
    }
    .metric-label {
        color: #6c757d; font-size: 0.85rem; margin-top: 0.3rem;
        text-transform: uppercase; font-weight: 500; letter-spacing: 1px;
    }

    .section-header {
        color: #2c5364; font-size: 1.5rem; font-weight: 600;
        margin: 2rem 0 1rem; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── 4. Helper Functions ────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """Generates a realistic synthetic HR dataset."""
    np.random.seed(101)
    n = 1200
    
    # Features
    age = np.random.normal(37, 9, n).astype(int)
    age = np.clip(age, 22, 60)
    
    departments = np.random.choice(["Sales", "Research & Development", "Human Resources", "IT"], n, p=[0.3, 0.45, 0.1, 0.15])
    
    distance = np.random.exponential(scale=10, size=n).astype(int)
    distance = np.clip(distance, 1, 30)
    
    education = np.random.choice(["High School", "Bachelor's", "Master's", "PhD"], n, p=[0.2, 0.5, 0.25, 0.05])
    
    job_satisfaction = np.random.choice([1, 2, 3, 4], n, p=[0.2, 0.2, 0.3, 0.3]) # 1=Low, 4=High
    
    monthly_income = np.where(education == "High School", np.random.normal(3000, 500, n),
                     np.where(education == "Bachelor's", np.random.normal(5000, 1000, n),
                     np.where(education == "Master's", np.random.normal(7500, 1500, n),
                              np.random.normal(10000, 2000, n))))
    monthly_income = np.clip(monthly_income, 2000, 20000).astype(int)
    
    years_at_company = np.clip(np.random.normal(7, 5, n), 0, 40).astype(int)
    
    # Target (Attrition probability based on features)
    attrition_prob = 0.15 # base rate
    attrition_prob += np.where(job_satisfaction <= 2, 0.2, -0.1)
    attrition_prob += np.where(monthly_income < 4000, 0.15, -0.05)
    attrition_prob += np.where(distance > 20, 0.1, 0)
    attrition_prob += np.where(years_at_company < 2, 0.1, -0.05)
    
    attrition_prob = np.clip(attrition_prob, 0, 1)
    attrition = np.random.binomial(1, attrition_prob)
    attrition_label = np.where(attrition == 1, "Yes", "No")
    
    df = pd.DataFrame({
        "Age": age,
        "Department": departments,
        "Distance_From_Home": distance,
        "Education": education,
        "Job_Satisfaction": job_satisfaction,
        "Monthly_Income": monthly_income,
        "Years_At_Company": years_at_company,
        "Attrition": attrition_label
    })
    return df

def prepare_features(df):
    df_ml = df.copy()
    encoders = {}
    cat_cols = ["Department", "Education", "Attrition"]
    for col in cat_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col])
        encoders[col] = le
    return df_ml, encoders

# ── 5. Load Data ───────────────────────────────────────────────────────────
df = load_data()

# ── 6. Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏢 HR Navigation")
    
    nav_options = [
        "📊 HR Dashboard",
        "🔍 EDA & Attrition Factors",
        "📋 Data Explorer",
    ]
    if _sklearn_available:
        nav_options.insert(2, "🤖 Attrition Prediction (ML)")
    
    page = st.radio("Go to", nav_options, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 🔧 Filters")
    dept_filter = st.multiselect("Department", df["Department"].unique(), default=df["Department"].unique())
    edu_filter  = st.multiselect("Education", df["Education"].unique(), default=df["Education"].unique())
    
    if not _sklearn_available:
        st.markdown("---")
        st.warning("⚠️ Machine Learning modules disabled due to system policy. Analytics features remain fully functional.")
        
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#6c757d;font-size:0.8rem'>Priya Ranjan Pandey<br>IBM SkillsBuild 2026</div>", unsafe_allow_html=True)

filtered_df = df[(df["Department"].isin(dept_filter)) & (df["Education"].isin(edu_filter))]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 — DASHBOARD                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if page == "📊 HR Dashboard":
    st.markdown(
        '<div class="main-header">'
        '<h1>🏢 Employee Attrition Dashboard</h1>'
        '<p>Analyzing HR metrics and identifying factors driving employee turnover</p>'
        '</div>',
        unsafe_allow_html=True
    )
    
    c1, c2, c3, c4 = st.columns(4)
    total_emp = len(filtered_df)
    attrition_count = len(filtered_df[filtered_df["Attrition"] == "Yes"])
    attrition_rate = (attrition_count / total_emp * 100) if total_emp > 0 else 0
    
    metrics = [
        (c1, f"{total_emp:,}", "Total Employees"),
        (c2, f"{attrition_rate:.1f}%", "Attrition Rate"),
        (c3, f"${filtered_df['Monthly_Income'].mean():,.0f}", "Avg Monthly Income"),
        (c4, f"{filtered_df['Years_At_Company'].mean():.1f}", "Avg Years at Company"),
    ]
    
    for col, val, label in metrics:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    st.markdown("")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">📈 Attrition Overview</p>', unsafe_allow_html=True)
        fig1 = px.pie(filtered_df, names="Attrition", hole=0.5, color="Attrition",
                      color_discrete_map={"Yes": "#e63946", "No": "#457b9d"})
        fig1.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.markdown('<p class="section-header">🏢 Attrition by Department</p>', unsafe_allow_html=True)
        dept_attr = filtered_df.groupby(["Department", "Attrition"]).size().reset_index(name="Count")
        fig2 = px.bar(dept_attr, x="Department", y="Count", color="Attrition", barmode="group",
                      color_discrete_map={"Yes": "#e63946", "No": "#457b9d"})
        fig2.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<p class="section-header">💰 Income vs Attrition</p>', unsafe_allow_html=True)
        fig3 = px.box(filtered_df, x="Attrition", y="Monthly_Income", color="Attrition",
                      color_discrete_map={"Yes": "#e63946", "No": "#457b9d"})
        fig3.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig3, use_container_width=True)
        
    with col4:
        st.markdown('<p class="section-header">😊 Job Satisfaction Levels</p>', unsafe_allow_html=True)
        sat_attr = filtered_df.groupby(["Job_Satisfaction", "Attrition"]).size().reset_index(name="Count")
        fig4 = px.bar(sat_attr, x="Job_Satisfaction", y="Count", color="Attrition", barmode="stack",
                      color_discrete_map={"Yes": "#e63946", "No": "#457b9d"})
        fig4.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), xaxis_title="Satisfaction (1=Low, 4=High)")
        st.plotly_chart(fig4, use_container_width=True)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 — EDA                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
elif page == "🔍 EDA & Attrition Factors":
    st.markdown('<div class="main-header"><h1>🔍 Deep Dive EDA</h1><p>Understanding what drives employees to leave</p></div>', unsafe_allow_html=True)
    
    st.markdown('<p class="section-header">📊 Correlation Heatmap</p>', unsafe_allow_html=True)
    df_ml, _ = prepare_features(filtered_df)
    corr = df_ml.corr()
    fig5 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
    fig5.update_layout(height=500)
    st.plotly_chart(fig5, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">🎂 Age Distribution</p>', unsafe_allow_html=True)
        fig6 = px.histogram(filtered_df, x="Age", color="Attrition", barmode="overlay", nbins=30,
                            color_discrete_map={"Yes": "#e63946", "No": "#457b9d"}, opacity=0.7)
        st.plotly_chart(fig6, use_container_width=True)
        
    with col2:
        st.markdown('<p class="section-header">🚗 Distance From Home</p>', unsafe_allow_html=True)
        fig7 = px.violin(filtered_df, x="Attrition", y="Distance_From_Home", color="Attrition",
                         box=True, color_discrete_map={"Yes": "#e63946", "No": "#457b9d"})
        st.plotly_chart(fig7, use_container_width=True)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 3 — ML PREDICTIONS                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
elif _sklearn_available and page == "🤖 Attrition Prediction (ML)":
    st.markdown('<div class="main-header"><h1>🤖 Machine Learning Predictions</h1><p>Predicting employee turnover using Classification Models</p></div>', unsafe_allow_html=True)
    
    df_ml, encoders = prepare_features(df)
    X = df_ml.drop("Attrition", axis=1)
    y = df_ml["Attrition"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(probability=True)
    }
    
    model_choice = st.selectbox("🧠 Select Classifier", list(models.keys()))
    
    if st.button("🚀 Train & Evaluate"):
        model = models[model_choice]
        with st.spinner("Training model..."):
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in [(c1, f"{acc:.1%}", "Accuracy"), (c2, f"{prec:.1%}", "Precision"), (c3, f"{rec:.1%}", "Recall"), (c4, f"{f1:.2f}", "F1 Score")]:
            col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-header">🧮 Confusion Matrix</p>', unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual"))
            fig_cm.update_layout(xaxis=dict(tickvals=[0,1], ticktext=["No", "Yes"]), yaxis=dict(tickvals=[0,1], ticktext=["No", "Yes"]))
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col2:
            if hasattr(model, "feature_importances_"):
                st.markdown('<p class="section-header">🔑 Feature Importance</p>', unsafe_allow_html=True)
                imp = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
                fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Viridis")
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 4 — DATA EXPLORER                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
elif page == "📋 Data Explorer":
    st.markdown('<div class="main-header"><h1>📋 Dataset Explorer</h1><p>View and download HR data</p></div>', unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True, height=500)
    st.download_button("📥 Download CSV", filtered_df.to_csv(index=False), "hr_attrition.csv", "text/csv")
