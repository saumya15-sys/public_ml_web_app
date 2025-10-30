# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 23:59:41 2025
@author: Saumya
Enhanced Health Assistant v2 - Cloud-compatible version
"""

import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import pandas as pd

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# ---------- STYLE ----------
st.markdown("""
    <style>
      body {background: linear-gradient(135deg, #f0f7ff 0%, #ffffff 100%);}
      .big-title {font-size:28px; color:#023e8a; font-weight:700;}
      .result-good {background:#e6ffed; border-left:6px solid #2d6a4f; padding:12px; border-radius:8px;}
      .result-bad {background:#ffecec; border-left:6px solid #9b2c2c; padding:12px; border-radius:8px;}
    </style>
""", unsafe_allow_html=True)

# ---------- FILE PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIABETES_MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.sav")
HEART_MODEL_PATH = os.path.join(BASE_DIR, "trained_model.sav")
HEALTH_IMAGE_PATH = os.path.join(BASE_DIR, "Health_image.png")

# ---------- LOAD MODELS ----------
def load_model(path):
    if not os.path.exists(path):
        st.warning(f"⚠️ Model file not found: `{os.path.basename(path)}`")
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

diabetes_model = load_model(DIABETES_MODEL_PATH)
heart_disease_model = load_model(HEART_MODEL_PATH)

# ---------- SIDEBAR ----------
with st.sidebar:
    selected = option_menu(
        '🧬 Multiple Disease Prediction System',
        ['🏥 Home', '💉 Diabetes Prediction', '❤️ Heart Disease Prediction'],
        menu_icon='hospital-fill',
        default_index=0
    )

# ---------- SESSION STATE ----------
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ---------- NORMALIZATION ----------
def _normalize(val, min_v, max_v):
    if val is None:
        return 0.0
    val = float(val)
    if val <= min_v:
        return 0.0
    if val >= max_v:
        return 1.0
    return (val - min_v) / (max_v - min_v)

# ---------- RISK SCORE FUNCTIONS ----------
def diabetes_risk_score(inputs):
    Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age = inputs
    w = {'Glucose':0.35, 'BMI':0.20, 'Age':0.12, 'DPF':0.15, 'Preg':0.06, 'Insulin':0.06, 'BP':0.04, 'Skin':0.02}
    score = (
        w['Glucose']*_normalize(Glucose, 50, 250) +
        w['BMI']*_normalize(BMI, 10, 60) +
        w['Age']*_normalize(Age, 10, 100) +
        w['DPF']*_normalize(DPF, 0.0, 2.5) +
        w['Preg']*_normalize(Pregnancies, 0, 20) +
        w['Insulin']*_normalize(Insulin, 0, 900) +
        w['BP']*_normalize(BP, 40, 160) +
        w['Skin']*_normalize(SkinThickness, 0, 100)
    )
    return min(max(score, 0.0), 1.0)

def heart_risk_score(inputs):
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = inputs
    w = {'chol':0.25, 'age':0.18, 'trestbps':0.12, 'thalach':0.15, 'oldpeak':0.12, 'ca':0.10, 'others':0.08}
    score = (
        w['chol']*_normalize(chol, 100, 600) +
        w['age']*_normalize(age, 10, 100) +
        w['trestbps']*_normalize(trestbps, 80, 220) +
        w['thalach']*(1.0 - _normalize(thalach, 60, 220)) +
        w['oldpeak']*_normalize(oldpeak, 0, 10) +
        w['ca']*_normalize(ca, 0, 3) +
        w['others']*((_normalize(cp,0,3)+_normalize(fbs,0,1)+_normalize(exang,0,1)+_normalize(restecg,0,2))/4.0)
    )
    return min(max(score, 0.0), 1.0)

# ---------- HOME ----------
if selected == '🏥 Home':
    st.markdown("""
        <div style="text-align:center; margin-top:-60px;">
            <h1 style="color:#023e8a; font-weight:800; font-size:42px;">
                Welcome to Health Assistant 🧑‍⚕️
            </h1>
            <p style="font-size:18px; color:#333333; margin-bottom:20px;">
                This web app predicts <b>Diabetes</b> and <b>Heart Disease</b> 
                using pre-trained ML models. Safe and interactive for everyone.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if os.path.exists(HEALTH_IMAGE_PATH):
        st.image(HEALTH_IMAGE_PATH, width="stretch")
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966482.png", width=360)

    st.markdown(
        "<div style='text-align:center; background-color:#e7f5ff; padding:10px; border-radius:8px; font-weight:500;'>"
        "Built with Streamlit • Developed by Saumya Arora"
        "</div>",
        unsafe_allow_html=True
    )

# ---------- DIABETES ----------
if selected == '💉 Diabetes Prediction':
    st.markdown('<div class="big-title">🩺 Diabetes Prediction</div>', unsafe_allow_html=True)

    with st.form(key='diab_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            pregnancies = st.number_input('Pregnancies', 0, 20, 0)
            skin_thickness = st.slider('Skin Thickness (mm)', 0, 100, 20)
            dpf = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.3725, 0.01)
        with col2:
            glucose = st.slider('Glucose Level (mg/dL)', 40, 250, 120)
            insulin = st.slider('Insulin Level (mu U/ml)', 0, 900, 80)
            age = st.number_input('Age', 1, 120, 30)
        with col3:
            bp = st.slider('Blood Pressure (mm Hg)', 40, 160, 70)
            bmi = st.slider('BMI (kg/m²)', 10.0, 60.0, 25.0, 0.1)
            submit_diab = st.form_submit_button('🔍 Get Diabetes Test Result')

    if submit_diab:
        user_input = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]
        if diabetes_model is not None:
            try:
                pred = diabetes_model.predict([user_input])[0]
                risk_pct = int(diabetes_risk_score(user_input) * 100)
                color = 'red' if pred == 1 else 'green'
                fig = px.bar(pd.DataFrame({'Metric':['Risk %'], 'Value':[risk_pct]}),
                             x='Value', y='Metric', orientation='h', range_x=[0,100],
                             text='Value', height=150)
                fig.update_traces(marker_color=color, textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                if pred == 1:
                    st.markdown('<div class="result-bad">🚨 The person is <b>Diabetic</b></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-good">✅ The person is <b>Not Diabetic</b></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("Diabetes model not loaded.")

# ---------- HEART ----------
if selected == '❤️ Heart Disease Prediction':
    st.markdown('<div class="big-title">💓 Heart Disease Prediction</div>', unsafe_allow_html=True)

    with st.form(key='heart_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input('Age', 1, 120, 45)
            trestbps = st.number_input('Resting BP (mm Hg)', 80, 250, 130)
            restecg = st.selectbox('Resting ECG', [0,1,2])
            oldpeak = st.slider('ST depression', 0.0, 10.0, 1.0, 0.1)
        with col2:
            sex = st.selectbox('Sex', ['Male','Female'])
            chol = st.number_input('Cholesterol (mg/dl)', 100, 600, 230)
            thalach = st.number_input('Max Heart Rate', 60, 220, 150)
            slope = st.selectbox('Slope of ST segment', [0,1,2])
        with col3:
            cp = st.selectbox('Chest Pain Type', [0,1,2,3])
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No','Yes'])
            exang = st.selectbox('Exercise Induced Angina', ['No','Yes'])
            ca = st.selectbox('Major Vessels (0-3)', [0,1,2,3])
            thal = st.selectbox('Thal', [0,1,2], format_func=lambda x: {0:'Normal',1:'Fixed Defect',2:'Reversible Defect'}[x])
        submit_heart = st.form_submit_button('🔍 Get Heart Disease Test Result')

    if submit_heart:
        sex_code = 1 if sex == 'Male' else 0
        fbs_code = 1 if fbs == 'Yes' else 0
        exang_code = 1 if exang == 'Yes' else 0
        user_input = [age, sex_code, cp, trestbps, chol, fbs_code, restecg,
                      thalach, exang_code, oldpeak, slope, ca, thal]

        if heart_disease_model is not None:
            try:
                pred = heart_disease_model.predict([user_input])[0]
                risk_pct = int(heart_risk_score(user_input) * 100)
                color = 'red' if pred == 1 else 'green'
                fig = px.bar(pd.DataFrame({'Metric':['Risk %'], 'Value':[risk_pct]}),
                             x='Value', y='Metric', orientation='h', range_x=[0,100],
                             text='Value', height=150)
                fig.update_traces(marker_color=color, textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                if pred == 1:
                    st.markdown('<div class="result-bad">🚨 The person <b>has Heart Disease</b></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-good">✅ The person <b>does not have Heart Disease</b></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("Heart model not loaded.")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("<div style='text-align:center;'>Built with ❤️ by Saumya Arora</div>", unsafe_allow_html=True)
