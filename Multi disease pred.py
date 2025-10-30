# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 23:59:41 2025
@author: mausa
Enhanced Health Assistant v2 - Dropdowns, validation, graphs (Plotly)
"""

import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import pandas as pd

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# ---------- STYLE ----------
st.markdown(
    """
    <style>
      body {background: linear-gradient(135deg, #f0f7ff 0%, #ffffff 100%);}
      .big-title {font-size:28px; color:#023e8a; font-weight:700;}
      .card {border-radius:10px; padding:12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);}
      .result-good {background:#e6ffed; border-left:6px solid #2d6a4f; padding:12px; border-radius:8px;}
      .result-bad {background:#ffecec; border-left:6px solid #9b2c2c; padding:12px; border-radius:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- LOAD MODELS ----------
# Update these paths if your model files are in another location
DIABETES_MODEL_PATH = r'C:/Users/mausa/OneDrive/Desktop/ML/Model/diabetes_model.sav'
HEART_MODEL_PATH = r'C:/Users/mausa/OneDrive/Desktop/ML/Model/trained_model.sav'

def load_model(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model from `{path}`: {e}")
        return None

diabetes_model = load_model(DIABETES_MODEL_PATH)
heart_disease_model = load_model(HEART_MODEL_PATH)

# ---------- SIDEBAR ----------
with st.sidebar:
    selected = option_menu('🧬 Multiple Disease Prediction System',
                           ['🏥 Home',
                            '💉 Diabetes Prediction',
                            '❤️ Heart Disease Prediction'],
                           menu_icon='hospital-fill',
                           default_index=0)

# ---------- SESSION STATE FOR HISTORY ----------
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ---------- UTILITIES: heuristic risk scorers ----------
def _normalize(val, min_v, max_v):
    if val is None:
        return 0.0
    val = float(val)
    if val <= min_v:
        return 0.0
    if val >= max_v:
        return 1.0
    return (val - min_v) / (max_v - min_v)

def diabetes_risk_score(inputs):
    # inputs order: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age = inputs
    # weights chosen to make a reasonable visualization (not model probability)
    w = {
        'Glucose': 0.35,
        'BMI': 0.20,
        'Age': 0.12,
        'DPF': 0.15,
        'Pregnancies': 0.06,
        'Insulin': 0.06,
        'BP': 0.04,
        'Skin': 0.02
    }
    s_gl = _normalize(Glucose, 50, 250)
    s_bmi = _normalize(BMI, 10, 60)
    s_age = _normalize(Age, 10, 100)
    s_dpf = _normalize(DPF, 0.0, 2.5)
    s_preg = _normalize(Pregnancies, 0, 20)
    s_ins = _normalize(Insulin, 0, 900)
    s_bp = _normalize(BloodPressure, 40, 160)
    s_skin = _normalize(SkinThickness, 0, 100)
    score = (w['Glucose']*s_gl + w['BMI']*s_bmi + w['Age']*s_age + w['DPF']*s_dpf +
             w['Pregnancies']*s_preg + w['Insulin']*s_ins + w['BP']*s_bp + w['Skin']*s_skin)
    # normalize to 0-1 (weights already sum ~1)
    return min(max(score, 0.0), 1.0)

def heart_risk_score(inputs):
    # inputs order: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = inputs
    w = {'chol':0.25, 'age':0.18, 'trestbps':0.12, 'thalach':0.15, 'oldpeak':0.12, 'ca':0.10, 'others':0.08}
    s_chol = _normalize(chol, 100, 600)
    s_age = _normalize(age, 10, 100)
    s_tbp = _normalize(trestbps, 80, 220)
    s_thalach = 1.0 - _normalize(thalach, 60, 220)  # lower max heart rate increases risk
    s_old = _normalize(oldpeak, 0, 10)
    s_ca = _normalize(ca, 0, 3)
    s_others = (_normalize(cp,0,3) + _normalize(fbs,0,1) + _normalize(exang,0,1) + _normalize(restecg,0,2))/4.0
    score = (w['chol']*s_chol + w['age']*s_age + w['trestbps']*s_tbp + w['thalach']*s_thalach +
             w['oldpeak']*s_old + w['ca']*s_ca + w['others']*s_others)
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
                using pre-trained ML models. Inputs are provided via dropdowns/sliders 
                so anyone can interact safely.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='text-align:center; background-color:#e7f5ff; padding:10px; border-radius:8px; font-weight:500;'>"
        "Built with Streamlit • Developed by Saumya Arora"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 🩺 High-quality, large, professional illustration
    st.image(
        "C:/Users/mausa/OneDrive/Desktop/ML/Multi disease prediction system/Health_image.png",
        use_container_width=True,
        caption="",
    )



   

# ---------- DIABETES PAGE ----------
if selected == '💉 Diabetes Prediction':
    st.markdown('<div class="big-title">🩺 Diabetes Prediction</div>', unsafe_allow_html=True)
    st.write("Fill the fields below. Values are limited to realistic ranges to help non-experts.")

    with st.form(key='diab_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0, step=1)
            skin_thickness = st.slider('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
            dpf = st.slider('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.3725, step=0.01)
        with col2:
            glucose = st.slider('Glucose Level (mg/dL)', min_value=40, max_value=250, value=120)
            insulin = st.slider('Insulin Level (mu U/ml)', min_value=0, max_value=900, value=80)
            age = st.number_input('Age', min_value=1, max_value=120, value=30)
        with col3:
            bp = st.slider('Blood Pressure (mm Hg)', min_value=40, max_value=160, value=70)
            bmi = st.slider('BMI (kg/m²)', min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            submit_diab = st.form_submit_button('🔍 Get Diabetes Test Result')

    if submit_diab:
        # validation (redundant since widgets restrict ranges, but kept for safety)
        try:
            user_input = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]
            user_input_f = [float(x) for x in user_input]
        except Exception:
            st.warning("Please make sure all inputs are numeric and within the provided ranges.")
            st.stop()

        if diabetes_model is None:
            st.error("Diabetes model not loaded. Check model path.")
        else:
            # model prediction (expects 0 or 1)
            try:
                pred = diabetes_model.predict([user_input_f])[0]
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                pred = None

            # heuristic risk score for visualization
            risk_score = diabetes_risk_score(user_input_f)
            risk_pct = int(risk_score * 100)

            # Build Plotly bar for risk
            df = pd.DataFrame({
                'Metric': ['Risk %'],
                'Value': [risk_pct]
            })
            color = 'red' if pred == 1 else 'green'
            fig = px.bar(df, x='Value', y='Metric', orientation='h', range_x=[0,100],
                         text='Value', height=150)
            fig.update_traces(marker_color=color, textposition='outside')
            fig.update_layout(margin=dict(l=20, r=20, t=10, b=10))

            # Result card
            if pred == 1:
                st.markdown('<div class="result-bad"><b>Result:</b> 🚨 The person is <b>Diabetic</b></div>', unsafe_allow_html=True)
                st.write(f"Risk visualization (heuristic): **{risk_pct}%**")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Advice:** Recommend a full medical check-up; consult a doctor. Lifestyle changes (diet, exercise) can help.")
            elif pred == 0:
                st.markdown('<div class="result-good"><b>Result:</b> ✅ The person is <b>Not Diabetic</b></div>', unsafe_allow_html=True)
                st.write(f"Risk visualization (heuristic): **{risk_pct}%**")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Advice:** Keep maintaining healthy habits; repeat screening as advised.")
            else:
                st.warning("Prediction unavailable.")

            # append to session history
            st.session_state.history.append({
                'type': 'Diabetes',
                'inputs': user_input_f,
                'prediction': 'Diabetic' if pred == 1 else 'Not Diabetic',
                'risk_pct': risk_pct
            })

# ---------- HEART PAGE ----------
if selected == '❤️ Heart Disease Prediction':
    st.markdown('<div class="big-title">💓 Heart Disease Prediction</div>', unsafe_allow_html=True)
    st.write("Choose options and slide values. Categorical fields use dropdowns (translated to numeric codes expected by the model).")

    with st.form(key='heart_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input('Age', min_value=1, max_value=120, value=45)
            trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=250, value=130)
            restecg = st.selectbox('Resting ECG', options=[0,1,2], index=0, format_func=lambda x: f"{x}")
            oldpeak = st.slider('ST depression (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        with col2:
            sex = st.selectbox('Sex', options=['Male','Female'])
            chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=230)
            thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
            slope = st.selectbox('Slope of ST segment', options=[0,1,2])
        with col3:
            cp = st.selectbox('Chest Pain Type', options=[0,1,2,3])
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=['No','Yes'])
            exang = st.selectbox('Exercise Induced Angina', options=['No','Yes'])
            ca = st.selectbox('Number of Major Vessels (0-3)', options=[0,1,2,3])
            thal = st.selectbox('Thal', options=[0,1,2], format_func=lambda x: {0:'Normal',1:'Fixed Defect',2:'Reversible Defect'}[x])
        submit_heart = st.form_submit_button('🔍 Get Heart Disease Test Result')

    if submit_heart:
        # map categorical to numeric codes the model expects
        sex_code = 1 if sex == 'Male' else 0
        fbs_code = 1 if fbs == 'Yes' else 0
        exang_code = 1 if exang == 'Yes' else 0

        try:
            user_input = [age, sex_code, cp, trestbps, chol, fbs_code, restecg,
                          thalach, exang_code, oldpeak, slope, ca, thal]
            user_input_f = [float(x) for x in user_input]
        except Exception:
            st.warning("Please check inputs.")
            st.stop()

        if heart_disease_model is None:
            st.error("Heart model not loaded. Check model path.")
        else:
            try:
                pred = heart_disease_model.predict([user_input_f])[0]
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                pred = None

            risk_score = heart_risk_score(user_input_f)
            risk_pct = int(risk_score * 100)

            df = pd.DataFrame({'Metric':['Risk %'],'Value':[risk_pct]})
            color = 'red' if pred == 1 else 'green'
            fig = px.bar(df, x='Value', y='Metric', orientation='h', range_x=[0,100],
                         text='Value', height=150)
            fig.update_traces(marker_color=color, textposition='outside')
            fig.update_layout(margin=dict(l=20, r=20, t=10, b=10))

            if pred == 1:
                st.markdown('<div class="result-bad"><b>Result:</b> 🚨 The person <b>has Heart Disease</b></div>', unsafe_allow_html=True)
                st.write(f"Risk visualization (heuristic): **{risk_pct}%**")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Advice:** Seek medical consultation. Consider ECG / stress testing as recommended by a physician.")
            elif pred == 0:
                st.markdown('<div class="result-good"><b>Result:</b> ✅ The person <b>does not have Heart Disease</b></div>', unsafe_allow_html=True)
                st.write(f"Risk visualization (heuristic): **{risk_pct}%**")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Advice:** Maintain healthy lifestyle and regular check-ups.")
            else:
                st.warning("Prediction unavailable.")

            st.session_state.history.append({
                'type': 'Heart',
                'inputs': user_input_f,
                'prediction': 'Heart Disease' if pred == 1 else 'No Heart Disease',
                'risk_pct': risk_pct
            })

# ---------- HISTORY PANEL ----------
if st.sidebar.checkbox("Show Prediction History", value=False):
    st.sidebar.markdown("### Prediction History")
    if len(st.session_state.history) == 0:
        st.sidebar.write("No predictions yet.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.sidebar.dataframe(hist_df[['type','prediction','risk_pct']].tail(10))

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("Built with ❤️ — Saumya Arora")
