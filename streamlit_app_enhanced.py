"""
CIBIL Score Predictor Pro - ULTIMATE VERSION
=============================================
Install:
    pip install streamlit xgboost shap pandas numpy scikit-learn plotly google-genai fpdf2 streamlit-authenticator bcrypt

Run:
    streamlit run streamlit_app_enhanced.py
"""

import os
import json
import io
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
from fpdf import FPDF

import asyncio
try:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())
except Exception:
    asyncio.set_event_loop(asyncio.new_event_loop())


# ── Paste your Gemini API key here ──────────────────────────────────────────
os.environ["GEMINI_API_KEY"] = st.secrets.get("GEMINI_API_KEY", "")


# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AUTOCRED",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "history"    not in st.session_state: st.session_state.history    = []
if "last_prediction" not in st.session_state: st.session_state.last_prediction = None
if "last_ai"         not in st.session_state: st.session_state.last_ai = None
if "theme"      not in st.session_state: st.session_state.theme      = "dark"
if "lang"       not in st.session_state: st.session_state.lang       = "English"
if "logged_in"  not in st.session_state: st.session_state.logged_in  = False
if "username"   not in st.session_state: st.session_state.username   = ""
if "users_db"   not in st.session_state: st.session_state.users_db   = {}
if "saved_data" not in st.session_state: st.session_state.saved_data = {}

# ─────────────────────────────────────────────────────────────────────────────
# TRANSLATIONS
# ─────────────────────────────────────────────────────────────────────────────
T = {
    "English": {
        "title":        "💳 AUTOCRED",
        "subtitle":     "XGBoost · SHAP · Gemini AI · Loan Eligibility · PDF Reports",
        "your_profile": "Your Credit Profile",
        "payment_hist": "Payment History (%)",
        "credit_util":  "Credit Utilization (0.0 to 1.0)",
        "credit_age":   "Credit Age (Years)",
        "num_accounts": "Number of Accounts",
        "hard_inq":     "Hard Inquiries",
        "dti":          "Debt-to-Income Ratio",
        "predict_btn":  "🚀 Predict My CIBIL Score",
        "model_perf":   "Model Performance",
        "clear_hist":   "Clear History",
        "pred_score":   "PREDICTED CIBIL SCORE",
        "range":        "Range: 300 to 900",
        "percentile":   "Percentile",
        "shap_tab":     "📊 SHAP",
        "radar_tab":    "🕸️ Radar",
        "tips_tab":     "💡 Tips",
        "ai_tab":       "🤖 AI Report",
        "loan_tab":     "🏦 Loans",
        "history_tab":  "📈 History",
        "whatif_tab":   "🔮 What-If",
        "emi_tab":      "🧮 EMI",
        "roadmap_tab":  "🗺️ Roadmap",
        "compare_tab":  "📊 Compare",
        "welcome":      "Welcome to AI-Driven Credit Analyzer",
        "login":        "Login",
        "register":     "Register",
        "logout":       "Logout",
        "username":     "Username",
        "password":     "Password",
        "save_data":    "Save My Data",
        "email_report": "Email Report",
        "send_email":   "Send Report to Email",
        "your_email":   "Your Email Address",
        "analyzing":    "Analyzing your credit profile...",
        "generating":   "Generating Gemini AI insights...",
    },
    "Hindi": {
        "title":        "💳 सिबिल स्कोर प्रेडिक्टर प्रो",
        "subtitle":     "XGBoost · SHAP · Gemini AI · लोन पात्रता · PDF रिपोर्ट",
        "your_profile": "आपकी क्रेडिट प्रोफाइल",
        "payment_hist": "भुगतान इतिहास (%)",
        "credit_util":  "क्रेडिट उपयोग (0.0 से 1.0)",
        "credit_age":   "क्रेडिट आयु (वर्ष)",
        "num_accounts": "खातों की संख्या",
        "hard_inq":     "हार्ड इन्क्वायरी",
        "dti":          "ऋण-से-आय अनुपात",
        "predict_btn":  "🚀 मेरा सिबिल स्कोर देखें",
        "model_perf":   "मॉडल प्रदर्शन",
        "clear_hist":   "इतिहास साफ करें",
        "pred_score":   "अनुमानित सिबिल स्कोर",
        "range":        "रेंज: 300 से 900",
        "percentile":   "प्रतिशतक",
        "shap_tab":     "📊 SHAP",
        "radar_tab":    "🕸️ रडार",
        "tips_tab":     "💡 सुझाव",
        "ai_tab":       "🤖 AI रिपोर्ट",
        "loan_tab":     "🏦 लोन",
        "history_tab":  "📈 इतिहास",
        "whatif_tab":   "🔮 क्या-अगर",
        "emi_tab":      "🧮 EMI",
        "roadmap_tab":  "🗺️ रोडमैप",
        "compare_tab":  "📊 तुलना",
        "welcome":      "सिबिल स्कोर प्रेडिक्टर प्रो में आपका स्वागत है",
        "login":        "लॉगिन",
        "register":     "रजिस्टर",
        "logout":       "लॉगआउट",
        "username":     "उपयोगकर्ता नाम",
        "password":     "पासवर्ड",
        "save_data":    "मेरा डेटा सेव करें",
        "email_report": "ईमेल रिपोर्ट",
        "send_email":   "ईमेल पर रिपोर्ट भेजें",
        "your_email":   "आपका ईमेल पता",
        "analyzing":    "आपकी क्रेडिट प्रोफाइल का विश्लेषण हो रहा है...",
        "generating":   "Gemini AI अंतर्दृष्टि उत्पन्न हो रही है...",
    },
}
def t(key):
    lang = st.session_state.get("lang", "English")
    return T.get(lang, T["English"]).get(key, key)

# ─────────────────────────────────────────────────────────────────────────────
# THEME CSS
# ─────────────────────────────────────────────────────────────────────────────
def get_css(theme):
    if theme == "dark":
        return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background: #060b18;
    background-image:
        radial-gradient(ellipse at 10% 0%, rgba(56,189,248,0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 100%, rgba(139,92,246,0.12) 0%, transparent 50%);
    color: #e2e8f0;
}
header[data-testid="stHeader"] { background: transparent !important; height: 0px !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 1280px !important; }
section[data-testid="stSidebar"] {
    width: 340px !important; min-width: 340px !important;
    background: rgba(8,15,35,0.95);
    border-right: 1px solid rgba(56,189,248,0.15);
    padding: 0 20px 30px 20px !important;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; font-family: 'Syne', sans-serif !important; }
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #7c3aed) !important;
    color: white !important; font-family: 'Syne', sans-serif !important;
    font-size: 16px !important; font-weight: 700 !important;
    padding: 14px 20px !important; border-radius: 14px !important;
    border: none !important; width: 100% !important;
    box-shadow: 0 4px 24px rgba(14,165,233,0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 32px rgba(14,165,233,0.5) !important; }
.score-hero { background: linear-gradient(135deg, rgba(14,165,233,0.15), rgba(124,58,237,0.15)); border: 1px solid rgba(56,189,248,0.3); border-radius: 24px; padding: 32px; text-align: center; }
.score-number { font-family: 'Syne', sans-serif; font-size: 80px; font-weight: 800; line-height: 1; margin: 0; }
.score-band { font-size: 16px; font-weight: 600; letter-spacing: 3px; margin-top: 6px; }
.section-title { font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 700; color: #f1f5f9 !important; margin: 0 0 14px 0; }
.metric-chip { background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.2); border-radius: 10px; padding: 12px 16px; text-align: center; }
.metric-chip .val { font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 700; color: #38bdf8; }
.metric-chip .lbl { font-size: 11px; color: #64748b; margin-top: 2px; letter-spacing: 1px; text-transform: uppercase; }
.tip-card { background: rgba(239,68,68,0.08); border-left: 3px solid #ef4444; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; font-size: 14px; color: #e2e8f0 !important; }
.tip-card-good { background: rgba(34,197,94,0.08); border-left: 3px solid #22c55e; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; font-size: 14px; color: #e2e8f0 !important; }
.ai-card { background: rgba(124,58,237,0.08); border: 1px solid rgba(124,58,237,0.25); border-radius: 16px; padding: 20px; margin-bottom: 14px; }
.ai-label { font-size: 10px; letter-spacing: 2px; color: #a78bfa; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; }
.rec-card { background: rgba(15,23,42,0.9); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 16px 18px; margin-bottom: 12px; display: flex; gap: 12px; align-items: flex-start; }
.rec-impact { padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; flex-shrink: 0; }
.impact-High { background: rgba(239,68,68,0.2); color: #f87171; }
.impact-Medium { background: rgba(251,191,36,0.2); color: #fbbf24; }
.impact-Low { background: rgba(34,197,94,0.2); color: #4ade80; }
.loan-card { border-radius: 16px; padding: 18px; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.08); }
.loan-eligible { background: rgba(34,197,94,0.08); border-left: 4px solid #22c55e; }
.loan-ineligible { background: rgba(239,68,68,0.08); border-left: 4px solid #ef4444; }
.loan-maybe { background: rgba(251,191,36,0.08); border-left: 4px solid #fbbf24; }
.history-card { background: rgba(15,23,42,0.8); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 14px 18px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
.whatif-card { background: rgba(56,189,248,0.06); border: 1px solid rgba(56,189,248,0.2); border-radius: 16px; padding: 20px; margin-bottom: 14px; }
.roadmap-step { background: rgba(15,23,42,0.8); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 16px 20px; margin-bottom: 12px; display: flex; gap: 16px; align-items: flex-start; }
.bank-card { background: rgba(15,23,42,0.8); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 16px; margin-bottom: 10px; }
.emi-result { background: linear-gradient(135deg, rgba(14,165,233,0.15), rgba(124,58,237,0.15)); border: 1px solid rgba(56,189,248,0.3); border-radius: 20px; padding: 28px; text-align: center; }
.login-box { background: rgba(15,23,42,0.9); border: 1px solid rgba(56,189,248,0.2); border-radius: 20px; padding: 32px; max-width: 400px; margin: 40px auto; }
.footer { text-align: center; color: #334155; margin-top: 50px; font-size: 13px; padding-bottom: 20px; }
.main-title { font-family: 'Syne', sans-serif; font-size: 48px; font-weight: 800; text-align: center; background: linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 4px; }
.sub-title { text-align: center; color: #475569; font-size: 16px; margin-bottom: 32px; }
.compare-bar { background: rgba(15,23,42,0.8); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; margin-bottom: 10px; }
div[data-baseweb="tab-list"] { background: rgba(15,23,42,0.8) !important; border-radius: 14px !important; padding: 4px !important; border: 1px solid rgba(255,255,255,0.08) !important; }
div[data-baseweb="tab"] { border-radius: 10px !important; font-weight: 500 !important; color: #64748b !important; }
div[aria-selected="true"][data-baseweb="tab"] { background: rgba(14,165,233,0.15) !important; color: #38bdf8 !important; }
</style>"""
    else:
        return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f0f4f8; color: #1e293b; }
header[data-testid="stHeader"] { background: transparent !important; height: 0px !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 1280px !important; }
section[data-testid="stSidebar"] {
    width: 340px !important; min-width: 340px !important;
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    padding: 0 20px 30px 20px !important;
}
section[data-testid="stSidebar"] * { color: #334155 !important; }
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #7c3aed) !important;
    color: white !important; font-family: 'Syne', sans-serif !important;
    font-size: 16px !important; font-weight: 700 !important;
    padding: 14px 20px !important; border-radius: 14px !important;
    border: none !important; width: 100% !important;
    box-shadow: 0 4px 24px rgba(14,165,233,0.25) !important;
}
.score-hero { background: linear-gradient(135deg, rgba(14,165,233,0.1), rgba(124,58,237,0.1)); border: 1px solid rgba(14,165,233,0.3); border-radius: 24px; padding: 32px; text-align: center; }
.score-number { font-family: 'Syne', sans-serif; font-size: 80px; font-weight: 800; line-height: 1; margin: 0; }
.score-band { font-size: 16px; font-weight: 600; letter-spacing: 3px; margin-top: 6px; }
.section-title { font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 700; color: #1e293b !important; margin: 0 0 14px 0; }
.metric-chip { background: rgba(14,165,233,0.08); border: 1px solid rgba(14,165,233,0.2); border-radius: 10px; padding: 12px 16px; text-align: center; }
.metric-chip .val { font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 700; color: #0ea5e9; }
.metric-chip .lbl { font-size: 11px; color: #94a3b8; margin-top: 2px; letter-spacing: 1px; text-transform: uppercase; }
.tip-card { background: rgba(239,68,68,0.06); border-left: 3px solid #ef4444; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; font-size: 14px; color: #1e293b !important; }
.tip-card-good { background: rgba(34,197,94,0.06); border-left: 3px solid #22c55e; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; font-size: 14px; color: #1e293b !important; }
.ai-card { background: rgba(124,58,237,0.05); border: 1px solid rgba(124,58,237,0.2); border-radius: 16px; padding: 20px; margin-bottom: 14px; }
.ai-label { font-size: 10px; letter-spacing: 2px; color: #7c3aed; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; }
.rec-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 16px 18px; margin-bottom: 12px; display: flex; gap: 12px; align-items: flex-start; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.rec-impact { padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; flex-shrink: 0; }
.impact-High { background: rgba(239,68,68,0.15); color: #dc2626; }
.impact-Medium { background: rgba(251,191,36,0.15); color: #d97706; }
.impact-Low { background: rgba(34,197,94,0.15); color: #16a34a; }
.loan-card { border-radius: 16px; padding: 18px; margin-bottom: 12px; border: 1px solid #e2e8f0; background: #ffffff; }
.loan-eligible { border-left: 4px solid #22c55e; }
.loan-ineligible { border-left: 4px solid #ef4444; }
.loan-maybe { border-left: 4px solid #fbbf24; }
.history-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 18px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
.whatif-card { background: rgba(14,165,233,0.05); border: 1px solid rgba(14,165,233,0.2); border-radius: 16px; padding: 20px; margin-bottom: 14px; }
.roadmap-step { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 16px 20px; margin-bottom: 12px; display: flex; gap: 16px; align-items: flex-start; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
.bank-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 16px; margin-bottom: 10px; }
.emi-result { background: linear-gradient(135deg, rgba(14,165,233,0.1), rgba(124,58,237,0.1)); border: 1px solid rgba(14,165,233,0.3); border-radius: 20px; padding: 28px; text-align: center; }
.login-box { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 20px; padding: 32px; max-width: 400px; margin: 40px auto; box-shadow: 0 4px 24px rgba(0,0,0,0.08); }
.footer { text-align: center; color: #94a3b8; margin-top: 50px; font-size: 13px; padding-bottom: 20px; }
.main-title { font-family: 'Syne', sans-serif; font-size: 48px; font-weight: 800; text-align: center; background: linear-gradient(90deg, #0ea5e9, #7c3aed, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 4px; }
.sub-title { text-align: center; color: #94a3b8; font-size: 16px; margin-bottom: 32px; }
div[data-baseweb="tab-list"] { background: #ffffff !important; border-radius: 14px !important; padding: 4px !important; border: 1px solid #e2e8f0 !important; }
div[data-baseweb="tab"] { border-radius: 10px !important; font-weight: 500 !important; color: #94a3b8 !important; }
div[aria-selected="true"][data-baseweb="tab"] { background: rgba(14,165,233,0.1) !important; color: #0ea5e9 !important; }
</style>"""

st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def score_band(s):
    if s >= 800: return "Exceptional", "#38bdf8"
    if s >= 750: return "Very Good",   "#34d399"
    if s >= 700: return "Good",        "#a3e635"
    if s >= 650: return "Fair",        "#fbbf24"
    return "Poor", "#f87171"

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def get_tip(feature):
    tips = {
        "Payment_History":      "Pay EMIs and credit card bills on time every month.",
        "Credit_Utilization":   "Keep credit utilization below 30% of your limit.",
        "Credit_Age":           "Maintain old accounts; avoid closing long-standing credit.",
        "Number_of_Accounts":   "Do not open many new accounts in a short span.",
        "Hard_Inquiries":       "Limit loan or card applications to avoid hard pulls.",
        "Debt_to_Income_Ratio": "Reduce existing debt or increase income to improve ratio.",
    }
    return tips.get(feature, "Maintain consistent financial discipline.")

# ─────────────────────────────────────────────────────────────────────────────
# USER AUTH
# ─────────────────────────────────────────────────────────────────────────────
def register_user(username, password):
    if username in st.session_state.users_db:
        return False, "Username already exists."
    st.session_state.users_db[username] = {
        "password": hash_pw(password),
        "history":  [],
        "created":  datetime.now().strftime("%d %b %Y"),
    }
    return True, "Registered successfully!"

def login_user(username, password):
    db = st.session_state.users_db
    if username not in db:
        return False, "Username not found."
    if db[username]["password"] != hash_pw(password):
        return False, "Wrong password."
    return True, "Login successful!"

# ─────────────────────────────────────────────────────────────────────────────
# LOAN ELIGIBILITY
# ─────────────────────────────────────────────────────────────────────────────
def get_loan_eligibility(score):
    loans = [
        {"name": "Home Loan",     "bank": "SBI / HDFC / ICICI",     "min_score": 750, "good_score": 800, "icon": "🏠", "rate": "8.5% - 9.5% p.a.",  "amount": "Up to ₹5 Crore"},
        {"name": "Car Loan",      "bank": "HDFC / Axis / Kotak",    "min_score": 700, "good_score": 750, "icon": "🚗", "rate": "9.0% - 11.0% p.a.", "amount": "Up to ₹50 Lakh"},
        {"name": "Personal Loan", "bank": "ICICI / Bajaj / IDFC",   "min_score": 700, "good_score": 750, "icon": "💰", "rate": "10.5% - 16.0% p.a.","amount": "Up to ₹40 Lakh"},
        {"name": "Credit Card",   "bank": "All Major Banks",         "min_score": 650, "good_score": 700, "icon": "💳", "rate": "2.5% - 3.5%/month", "amount": "Based on income"},
        {"name": "Business Loan", "bank": "SBI / HDFC / Yes Bank",  "min_score": 700, "good_score": 750, "icon": "🏢", "rate": "11.0% - 16.0% p.a.","amount": "Up to ₹2 Crore"},
        {"name": "Education Loan","bank": "SBI / Bank of Baroda",   "min_score": 650, "good_score": 700, "icon": "🎓", "rate": "8.0% - 11.5% p.a.", "amount": "Up to ₹1.5 Crore"},
    ]
    results = []
    for loan in loans:
        if score >= loan["good_score"]:
            status, label, color = "eligible",   "Eligible - Best Rates",             "#22c55e"
        elif score >= loan["min_score"]:
            status, label, color = "maybe",      "Eligible - Higher Rates",           "#fbbf24"
        else:
            status, label, color = "ineligible", f"Not Eligible (Need {loan['min_score']}+)", "#f87171"
        results.append({**loan, "status": status, "label": label, "color": color})
    return results

# ─────────────────────────────────────────────────────────────────────────────
# BANK-WISE LOAN OFFERS
# ─────────────────────────────────────────────────────────────────────────────
def get_bank_offers(score):
    all_offers = [
        {"bank": "SBI",          "logo": "🏛️", "product": "Home Loan",     "rate": "8.50%", "min_score": 750, "processing": "0.35%", "special": "Women get 0.05% discount"},
        {"bank": "HDFC Bank",    "logo": "🏦", "product": "Home Loan",     "rate": "8.70%", "min_score": 750, "processing": "0.50%", "special": "Pre-approved offers available"},
        {"bank": "ICICI Bank",   "logo": "🏧", "product": "Personal Loan", "rate": "10.75%","min_score": 720, "processing": "1.00%", "special": "Instant disbursal in 3 hours"},
        {"bank": "Axis Bank",    "logo": "💼", "product": "Car Loan",      "rate": "9.10%", "min_score": 700, "processing": "0.75%", "special": "100% on-road funding"},
        {"bank": "Kotak Mahindra","logo":"🏪", "product": "Personal Loan", "rate": "10.99%","min_score": 720, "processing": "1.00%", "special": "Zero foreclosure charges"},
        {"bank": "Bajaj Finserv","logo": "⚡", "product": "Personal Loan", "rate": "11.00%","min_score": 700, "processing": "1.50%", "special": "Flexi loan facility available"},
        {"bank": "IDFC First",   "logo": "🌟", "product": "Personal Loan", "rate": "10.49%","min_score": 700, "processing": "1.00%", "special": "No prepayment charges"},
        {"bank": "Yes Bank",     "logo": "✅", "product": "Business Loan", "rate": "11.50%","min_score": 700, "processing": "1.25%", "special": "Collateral-free up to ₹50L"},
    ]
    return [o for o in all_offers if score >= o["min_score"]]

# ─────────────────────────────────────────────────────────────────────────────
# SCORE IMPROVEMENT ROADMAP
# ─────────────────────────────────────────────────────────────────────────────
def get_roadmap(score, suggestions):
    target = 750 if score < 750 else 800 if score < 800 else 850
    steps = []

    step_num = 1
    if any("Payment" in s[0] for s in suggestions):
        steps.append({
            "step": step_num, "timeline": "Month 1-2", "icon": "📅",
            "title": "Set Up Auto-Pay",
            "detail": "Enable auto-debit for all EMIs and credit card minimum payments. Even one missed payment drops score by 50-100 points.",
            "impact": "+40 to +60 points", "color": "#ef4444",
        })
        step_num += 1

    if any("Utilization" in s[0] for s in suggestions):
        steps.append({
            "step": step_num, "timeline": "Month 1-3", "icon": "💳",
            "title": "Reduce Credit Utilization to Below 30%",
            "detail": "Pay down existing balances or request a credit limit increase. Aim for under 10% for maximum score boost.",
            "impact": "+30 to +50 points", "color": "#f97316",
        })
        step_num += 1

    if any("Inquiries" in s[0] for s in suggestions):
        steps.append({
            "step": step_num, "timeline": "Month 1-6", "icon": "🔍",
            "title": "Stop New Credit Applications",
            "detail": "Every hard inquiry reduces score by 5-10 points. Avoid applying for new loans or credit cards for at least 6 months.",
            "impact": "+15 to +25 points", "color": "#eab308",
        })
        step_num += 1

    if any("Debt" in s[0] for s in suggestions):
        steps.append({
            "step": step_num, "timeline": "Month 3-12", "icon": "💰",
            "title": "Reduce Debt-to-Income Ratio",
            "detail": "Focus on paying off high-interest loans first (avalanche method). Try to bring DTI below 35%.",
            "impact": "+20 to +35 points", "color": "#22c55e",
        })
        step_num += 1

    steps.append({
        "step": step_num, "timeline": "Month 6-12", "icon": "🏦",
        "title": "Diversify Credit Mix",
        "detail": "Having a mix of secured (home/auto) and unsecured (credit card/personal) loans improves score. Don't close old accounts.",
        "impact": "+10 to +20 points", "color": "#3b82f6",
    })

    steps.append({
        "step": step_num + 1, "timeline": "Month 12-24", "icon": "📈",
        "title": f"Reach Target Score of {target}",
        "detail": f"By consistently following all steps above, you can realistically reach {target}+ within 12-24 months.",
        "impact": f"Target: {target}+", "color": "#38bdf8",
    })

    return steps, target
# ─────────────────────────────────────────────────────────────────────────────
# EMI CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
def calculate_emi(principal, annual_rate, tenure_months):
    r = annual_rate / (12 * 100)
    if r == 0:
        return principal / tenure_months, principal, 0
    emi = principal * r * (1 + r)**tenure_months / ((1 + r)**tenure_months - 1)
    total = emi * tenure_months
    interest = total - principal
    return round(emi, 2), round(total, 2), round(interest, 2)

# ─────────────────────────────────────────────────────────────────────────────
# EMAIL REPORT
# ─────────────────────────────────────────────────────────────────────────────
def send_email_report(to_email, pdf_bytes, score, band):
    try:
        import base64
        import urllib.request

        api_key = st.secrets.get("BREVO_API_KEY", "")

        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        payload = json.dumps({
            "sender": {"name": "CIBIL Score Predictor", "email": "munashirafarheen@gmail.com"},
            "to": [{"email": to_email}],
            "subject": f"Your CIBIL Score Report - {score} ({band})",
            "htmlContent": f"""
                <h2 style="color:#0ea5e9;">Your CIBIL Score Report</h2>
                <p>Dear User,</p>
                <p>Please find attached your CIBIL Score Report.</p>
                <h1 style="color:#0ea5e9;">{score} <span style="font-size:18px;">({band})</span></h1>
                <p>Generated on: {datetime.now().strftime('%d %B %Y at %I:%M %p')}</p>
                <hr>
                <p style="color:#64748b;font-size:12px;">CIBIL Score Predictor Pro | For educational purposes only</p>
            """,
            "attachment": [{
                "content": pdf_base64,
                "name": f"CIBIL_Report_{score}.pdf"
            }]
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.brevo.com/v3/smtp/email",
            data=payload,
            headers={
                "api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            method="POST"
        )
        with urllib.request.urlopen(req) as response:
            if response.status == 201:
                return True, "✅ Email sent successfully!"
            else:
                return False, f"Failed with status: {response.status}"

    except Exception as e:
        return False, f"Email error: {str(e)}"

# ─────────────────────────────────────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf(score, band, ph, cu, ca, na, hi, dr, suggestions, positives, ai):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(6, 11, 24)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_text_color(56, 189, 248)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(10, 10)
    pdf.cell(190, 10, "AutoCred", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.set_xy(10, 25)
    pdf.cell(190, 8, f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(20)

    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(190, 10, "Predicted CIBIL Score", ln=True)
    pdf.set_font("Helvetica", "B", 48)
    pdf.set_text_color(14, 165, 233)
    pdf.cell(190, 20, str(score), align="C", ln=True)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(190, 8, f"Band: {band}   |   Range: 300 - 900", align="C", ln=True)
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(190, 10, "Credit Profile", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(71, 85, 105)
    for label, value in [("Payment History", f"{ph}%"), ("Credit Utilization", f"{round(cu*100,1)}%"),
                         ("Credit Age", f"{ca} years"), ("Number of Accounts", str(na)),
                         ("Hard Inquiries", str(hi)), ("Debt-to-Income Ratio", f"{round(dr*100,1)}%")]:
        pdf.cell(100, 8, label)
        pdf.cell(90, 8, value, ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(190, 10, "Strengths", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(34, 197, 94)
    for feat, val in positives:
        pdf.cell(190, 7, f"+ {feat.replace('_',' ')} (SHAP: +{val})", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(190, 10, "Areas to Improve", ln=True)
    for feat, tip, val in suggestions:
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(239, 68, 68)
        pdf.cell(190, 7, f"- {feat.replace('_',' ')} (SHAP: {val})", ln=True)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(190, 6, f"  Tip: {tip}", ln=True)
    pdf.ln(4)

    if ai:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(190, 10, "AI Credit Analysis", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(71, 85, 105)
        pdf.multi_cell(190, 7, ai.get("summary", ""))
        pdf.ln(3)
        for rec in ai.get("recommendations", []):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(14, 165, 233)
            pdf.cell(190, 7, f"[{rec.get('impact','').upper()}] {rec.get('title','')}", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(71, 85, 105)
            pdf.multi_cell(190, 6, f"  {rec.get('detail','')} ({rec.get('timeline','')})")

    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(190, 8, "AutoCred  |  For educational purposes only", align="C")
    return bytes(pdf.output())

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv("synthetic_cibil_scores.csv")
    features = ["Payment_History","Credit_Utilization","Credit_Age","Number_of_Accounts","Hard_Inquiries","Debt_to_Income_Ratio"]
    X = df[features].fillna(df[features].mean())
    y = df["CIBIL_Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = xgb.XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
    mdl.fit(X_train, y_train)
    explainer = shap.Explainer(mdl, X_train)
    y_pred = mdl.predict(X_test)
    metrics = {"R2 Score": round(r2_score(y_test, y_pred), 4), "MAE": round(mean_absolute_error(y_test, y_pred), 2), "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2)}
    return mdl, explainer, metrics

def predict_cibil(mdl, expl, values):
    cols = ["Payment_History","Credit_Utilization","Credit_Age","Number_of_Accounts","Hard_Inquiries","Debt_to_Income_Ratio"]
    df_in = pd.DataFrame([dict(zip(cols, values))])
    score = int(max(300, min(900, float(mdl.predict(df_in)[0]))))
    sv = expl(df_in)
    return score, df_in, sv

def get_suggestions(df_in, sv):
    neg, pos = [], []
    for i, val in enumerate(sv.values[0]):
        feat = df_in.columns[i]
        if val < 0: neg.append((feat, get_tip(feat), round(float(val), 2)))
        else:       pos.append((feat, round(float(val), 2)))
    return neg, pos

# ─────────────────────────────────────────────────────────────────────────────
# AI
# ─────────────────────────────────────────────────────────────────────────────
def get_ai_insights(score, ph, cu, ca, na, hi, dr, shap_vals):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key: return None
    try:
        from google import genai
    except ImportError:
        st.warning("Run: pip install google-genai")
        return None

    band, _ = score_band(score)
    shap_dict = {k: round(float(v), 2) for k, v in zip(
        ["Payment_History","Credit_Utilization","Credit_Age","Number_of_Accounts","Hard_Inquiries","Debt_to_Income_Ratio"],
        shap_vals)}

    prompt = (
        "You are a senior credit analyst in India. Analyze this CIBIL profile:\n"
        f"Payment History: {ph}%\nCredit Utilization: {round(cu*100,1)}%\nCredit Age: {ca} years\n"
        f"Number of Accounts: {na}\nHard Inquiries: {hi}\nDebt-to-Income Ratio: {round(dr*100,1)}%\n"
        f"Predicted CIBIL Score: {score} ({band})\nSHAP impacts: {json.dumps(shap_dict)}\n\n"
        "Respond ONLY with valid JSON (no markdown):\n"
        '{"summary":"3 sentence summary","strengths":["s1","s2"],"risks":["r1","r2"],'
        '"recommendations":[{"title":"...","detail":"...","impact":"High","timeline":"3-6 months"},'
        '{"title":"...","detail":"...","impact":"Medium","timeline":"6-12 months"},'
        '{"title":"...","detail":"...","impact":"Low","timeline":"1-2 years"}],'
        '"projection":{"3mo":' + str(min(900,score+8)) + ',"6mo":' + str(min(900,score+20)) +
        ',"12mo":' + str(min(900,score+40)) + ',"24mo":' + str(min(900,score+65)) + '}}'
    )
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model="gemma-3-4b-it", contents=prompt)
        raw = response.text.strip().replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except Exception as e:
        st.warning(f"AI insights error: {e}")
        return None
    # ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
BG = "rgba(0,0,0,0)"
GRID = "#1e293b" if st.session_state.theme == "dark" else "#e2e8f0"
TICK = "#64748b"

def gauge_chart(score):
    _, color = score_band(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"font": {"size": 52, "color": color, "family": "Syne"}},
        gauge={"axis": {"range": [300,900], "tickvals": [300,500,580,670,750,800,900],
                        "ticktext": ["300","500","Poor","Fair","Good","V.Good","900"],
                        "tickfont": {"color": TICK, "size": 10}},
               "bar": {"color": color, "thickness": 0.28}, "bgcolor": BG, "borderwidth": 0,
               "steps": [{"range":[300,580],"color":"rgba(239,68,68,0.12)"},{"range":[580,670],"color":"rgba(251,191,36,0.12)"},
                         {"range":[670,750],"color":"rgba(163,230,53,0.12)"},{"range":[750,800],"color":"rgba(52,211,153,0.12)"},
                         {"range":[800,900],"color":"rgba(56,189,248,0.12)"}],
               "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": score}}))
    fig.update_layout(height=260, margin=dict(l=20,r=20,t=20,b=10), paper_bgcolor=BG, plot_bgcolor=BG, font={"color":"#e2e8f0"})
    return fig

def shap_bar_chart(df_in, sv):
    vals = sv.values[0]
    feats = df_in.columns.tolist()
    idx = np.argsort(np.abs(vals))[::-1]
    fig = go.Figure(go.Bar(
        x=[float(vals[i]) for i in idx], y=[feats[i].replace("_"," ") for i in idx],
        orientation="h", marker_color=["#38bdf8" if vals[i]>0 else "#f87171" for i in idx],
        text=[f"{float(vals[i]):+.1f}" for i in idx], textfont={"color":"#e2e8f0","size":12}, textposition="outside"))
    fig.update_layout(height=300, margin=dict(l=10,r=50,t=20,b=10), paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis={"gridcolor":GRID,"zerolinecolor":GRID,"tickfont":{"color":TICK}},
        yaxis={"gridcolor":GRID,"tickfont":{"color":"#cbd5e1","size":13}}, showlegend=False)
    return fig

def radar_chart(df_in):
    factors = {
        "Payment History": float(df_in["Payment_History"].iloc[0]),
        "Credit Score":    max(0.0, 100.0 - float(df_in["Credit_Utilization"].iloc[0])*100),
        "Credit Age":      min(100.0, float(df_in["Credit_Age"].iloc[0])*2.5),
        "Account Mix":     min(100.0, float(df_in["Number_of_Accounts"].iloc[0])*10),
        "New Credit":      max(0.0, 100.0 - float(df_in["Hard_Inquiries"].iloc[0])*12),
        "Debt Ratio":      max(0.0, 100.0 - float(df_in["Debt_to_Income_Ratio"].iloc[0])*100),
    }
    labels = list(factors.keys()); values = list(factors.values())
    fig = go.Figure(go.Scatterpolar(r=values+[values[0]], theta=labels+[labels[0]],
        fill="toself", fillcolor="rgba(56,189,248,0.12)", line={"color":"#38bdf8","width":2}, marker={"color":"#38bdf8","size":6}))
    fig.update_layout(polar={"radialaxis":{"visible":True,"range":[0,100],"gridcolor":GRID,"tickfont":{"color":TICK,"size":10},"linecolor":GRID},
        "angularaxis":{"gridcolor":GRID,"tickfont":{"color":"#94a3b8","size":12},"linecolor":GRID},"bgcolor":BG},
        height=300, margin=dict(l=40,r=40,t=30,b=30), paper_bgcolor=BG, showlegend=False)
    return fig

def projection_chart(current_score, proj):
    x = ["Now","3 Months","6 Months","12 Months","24 Months"]
    y = [current_score, proj.get("3mo",current_score), proj.get("6mo",current_score), proj.get("12mo",current_score), proj.get("24mo",current_score)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers+text",
        line={"color":"#38bdf8","width":3,"shape":"spline"}, marker={"size":9,"color":"#38bdf8"},
        fill="tozeroy", fillcolor="rgba(56,189,248,0.07)",
        text=[str(int(v)) for v in y], textposition="top center", textfont={"color":"#38bdf8","size":12}))
    fig.update_layout(height=260, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis={"gridcolor":GRID,"tickfont":{"color":TICK},"linecolor":GRID},
        yaxis={"gridcolor":GRID,"tickfont":{"color":TICK},"linecolor":GRID,"range":[max(300,current_score-30),min(900,max(y)+30)]},
        showlegend=False)
    return fig

def distribution_chart(score):
    x = np.linspace(300, 900, 200)
    y = np.exp(-0.5 * ((x - 680) / 100) ** 2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, fill="tozeroy", fillcolor="rgba(100,116,139,0.15)", line={"color":GRID,"width":1.5}))
    fig.add_vline(x=score, line_width=2.5, line_color="#38bdf8",
        annotation_text=f"  Your score: {score}", annotation_font_color="#38bdf8", annotation_font_size=13)
    fig.update_layout(height=200, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis={"gridcolor":GRID,"tickfont":{"color":TICK},"title_text":"CIBIL Score"}, yaxis={"visible":False}, showlegend=False)
    return fig

def history_chart(history):
    dates = [h["time"] for h in history]; scores = [h["score"] for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=scores, mode="lines+markers",
        line={"color":"#38bdf8","width":2,"shape":"spline"},
        marker={"size":10,"color":[score_band(s)[1] for s in scores],"line":{"color":"#0f172a","width":2}},
        fill="tozeroy", fillcolor="rgba(56,189,248,0.06)"))
    fig.update_layout(height=250, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis={"gridcolor":GRID,"tickfont":{"color":TICK}},
        yaxis={"gridcolor":GRID,"tickfont":{"color":TICK},"range":[300,900]}, showlegend=False)
    return fig

def whatif_chart(original_score, whatif_score):
    fig = go.Figure(go.Bar(
        x=["Current Score","What-If Score"], y=[original_score, whatif_score],
        marker_color=[score_band(original_score)[1], score_band(whatif_score)[1]],
        text=[str(original_score), str(whatif_score)], textposition="outside",
        textfont={"color":"#e2e8f0","size":14}, width=0.4))
    fig.update_layout(height=280, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis={"tickfont":{"color":"#94a3b8","size":13},"gridcolor":GRID},
        yaxis={"gridcolor":GRID,"tickfont":{"color":TICK},"range":[300,950]}, showlegend=False)
    return fig

def compare_chart(your_score):
    avg_india = 720; good_score = 750
    fig = go.Figure()
    categories = ["Your Score", "Avg Indian\nCIBIL (720)", "Good Score\nTarget (750)"]
    values      = [your_score,   720,                        750]
    colors      = [score_band(your_score)[1], "#94a3b8", "#34d399"]
    fig.add_trace(go.Bar(x=categories, y=values, marker_color=colors,
        text=[str(v) for v in values], textposition="outside",
        textfont={"color":"#e2e8f0","size":14}, width=0.4))
    fig.add_hline(y=750, line_dash="dash", line_color="#34d399", annotation_text="Good Score Threshold",
        annotation_font_color="#34d399")
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis={"tickfont":{"color":"#94a3b8","size":12},"gridcolor":GRID},
        yaxis={"gridcolor":GRID,"tickfont":{"color":TICK},"range":[300,950]}, showlegend=False)
    return fig

def emi_breakdown_chart(principal, interest):
    fig = go.Figure(go.Pie(
        labels=["Principal", "Total Interest"],
        values=[principal, interest],
        hole=0.6,
        marker_colors=["#38bdf8","#f87171"],
        textfont={"color":"white","size":13}))
    fig.update_layout(height=260, margin=dict(l=10,r=10,t=20,b=10),
        paper_bgcolor=BG, showlegend=True,
        legend={"font":{"color":"#94a3b8"}})
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
model, explainer, model_metrics = train_model()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="main-title">{t("title")}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">{t("subtitle")}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # Theme + Language toggles
    col_t, col_l = st.columns(2)
    with col_t:
        if st.button("🌙 Dark" if st.session_state.theme == "light" else "☀️ Light"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
    with col_l:
        lang = st.selectbox("🌐", ["English","Hindi"], label_visibility="collapsed",
                            index=0 if st.session_state.lang=="English" else 1)
        if lang != st.session_state.lang:
            st.session_state.lang = lang
            st.rerun()

    st.markdown("---")

    # Login / Register
    if not st.session_state.logged_in:
        st.markdown("#### 👤 Account")
        auth_tab = st.radio("", ["Login","Register"], horizontal=True, label_visibility="collapsed")
        uname = st.text_input(t("username"), key="auth_uname")
        pword = st.text_input(t("password"), type="password", key="auth_pw")
        if auth_tab == "Login":
            if st.button(t("login")):
                ok, msg = login_user(uname, pword)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username  = uname
                    if uname in st.session_state.users_db:
                        st.session_state.history = st.session_state.users_db[uname].get("history", [])
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            if st.button(t("register")):
                ok, msg = register_user(uname, pword)
                st.success(msg) if ok else st.error(msg)
    else:
        st.markdown(f"#### 👤 {st.session_state.username}")
        if st.button(t("logout")):
            if st.session_state.username in st.session_state.users_db:
                st.session_state.users_db[st.session_state.username]["history"] = st.session_state.history
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.rerun()
        if st.button(t("save_data")):
            if st.session_state.username in st.session_state.users_db:
                st.session_state.users_db[st.session_state.username]["history"] = st.session_state.history
                st.success("Data saved!")

    st.markdown("---")
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(14,165,233,0.15),rgba(124,58,237,0.15));
                border:1px solid rgba(56,189,248,0.3);border-radius:16px;
                padding:16px;margin-bottom:16px;text-align:center;">
        <div style="font-size:28px;">💳</div>
        <div style="font-size:15px;font-weight:700;color:#f1f5f9;margin-top:4px;">Your Credit Profile</div>
        <div style="font-size:11px;color:#475569;margin-top:2px;">Adjust sliders below</div>
    </div>
    """, unsafe_allow_html=True)

    ph = st.slider("💰 Payment History (%)",  0, 100, 80)
    cu = st.slider("📊 Credit Utilization",   0.0, 1.0, 0.30, step=0.01)
    ca = st.slider("📅 Credit Age (Years)",   0, 50, 10)
    na = st.slider("🏦 Number of Accounts",   1, 100, 8)
    hi = st.slider("🔍 Hard Inquiries",       0, 10, 2)
    dr = st.slider("💸 Debt-to-Income Ratio", 0.0, 1.0, 0.25, step=0.01)

    good_count = sum([ph>=80, cu<=0.3, ca>=10, 5<=na<=15, hi<=2, dr<=0.35])
    health_color = "#34d399" if good_count >= 5 else "#fbbf24" if good_count >= 3 else "#f87171"
    health_label = "Excellent 🌟" if good_count >= 5 else "Average ⚡" if good_count >= 3 else "Needs Work ⚠️"
    st.markdown(f"""
    <div style="background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.08);
                border-radius:12px;padding:12px;margin:8px 0;text-align:center;">
        <div style="font-size:11px;color:#475569;letter-spacing:1px;">PROFILE HEALTH</div>
        <div style="font-size:18px;font-weight:700;color:{health_color};">
            {good_count}/6 — {health_label}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    predict_btn = st.button(t("predict_btn"))

    st.markdown("---")
    for k, v in model_metrics.items():
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:6px 0;
                    border-bottom:1px solid rgba(255,255,255,0.05);">
            <span style="font-size:12px;color:#64748b;">{k}</span>
            <span style="font-size:12px;font-weight:700;color:#38bdf8;">{v}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button(t("clear_hist")):
        st.session_state.history = []
        st.rerun()
        # ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if predict_btn or st.session_state.last_prediction:
    if predict_btn:
        with st.spinner(t("analyzing")):
            score, df_input, shap_values = predict_cibil(model, explainer, [ph, cu, ca, na, hi, dr])
            suggestions, positives = get_suggestions(df_input, shap_values)
            band, color = score_band(score)
            percentile = int(((score - 300) / 600) * 100)

        with st.spinner(t("generating")):
            ai = get_ai_insights(score, ph, cu, ca, na, hi, dr, shap_values.values[0])

        st.session_state.last_prediction = (score, df_input, shap_values, suggestions, positives, band, color, percentile)
        st.session_state.last_ai = ai

        st.session_state.history.append({
            "time": datetime.now().strftime("%d %b %H:%M"),
            "score": score, "band": band,
            "ph": ph, "cu": cu, "ca": ca, "na": na, "hi": hi, "dr": dr,
        })

    else:
        score, df_input, shap_values, suggestions, positives, band, color, percentile = st.session_state.last_prediction
        ai = st.session_state.last_ai

    # Row 1
    col_score, col_gauge, col_metrics = st.columns([1, 1.2, 1])
    with col_score:
        st.markdown(f"""
        <div class="score-hero">
            <div style="font-size:12px;letter-spacing:2px;color:#64748b;margin-bottom:8px;">{t('pred_score')}</div>
            <div class="score-number" style="color:{color};">{score}</div>
            <div class="score-band" style="color:{color};">{band.upper()}</div>
            <div style="margin-top:16px;font-size:13px;color:#64748b;">{t('range')} &nbsp;|&nbsp; {t('percentile')}: {percentile}th</div>
        </div>""", unsafe_allow_html=True)
    with col_gauge:
        st.plotly_chart(gauge_chart(score), use_container_width=True, config={"displayModeBar":False})
    with col_metrics:
        dti_pct = round(dr*100,1); cu_pct = round(cu*100,1)
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px;">
            <div class="metric-chip"><div class="val">{ph}%</div><div class="lbl">Payment</div></div>
            <div class="metric-chip"><div class="val">{cu_pct}%</div><div class="lbl">Utilization</div></div>
            <div class="metric-chip"><div class="val">{ca}yr</div><div class="lbl">Credit Age</div></div>
            <div class="metric-chip"><div class="val">{dti_pct}%</div><div class="lbl">DTI Ratio</div></div>
            <div class="metric-chip" style="grid-column:span 2;">
                <div class="val">{na} accts &nbsp; {hi} inq.</div>
                <div class="lbl">Accounts and Inquiries</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10 = st.tabs([
        t("shap_tab"), t("radar_tab"), t("tips_tab"), t("ai_tab"),
        t("loan_tab"), t("history_tab"), t("whatif_tab"),
        t("emi_tab"), t("roadmap_tab"), t("compare_tab"),
    ])

    # ── Tab 1: SHAP ───────────────────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-title">Feature Impact on Score</p>', unsafe_allow_html=True)
            st.plotly_chart(shap_bar_chart(df_input, shap_values), use_container_width=True, config={"displayModeBar":False})
            st.caption("Blue = boosts score   |   Red = reduces score")
        with c2:
            st.markdown('<p class="section-title">Your Score vs Population</p>', unsafe_allow_html=True)
            st.plotly_chart(distribution_chart(score), use_container_width=True, config={"displayModeBar":False})
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-title">SHAP Waterfall</p>', unsafe_allow_html=True)
            try:
                    plt.clf()
                    plt.close("all")
                    bg_col = "#0f172a" if st.session_state.theme=="dark" else "#ffffff"
                    shap_fig, ax = plt.subplots(figsize=(10, 6))
                    shap_fig.patch.set_facecolor(bg_col)
                    ax.set_facecolor(bg_col)
                    shap.plots.waterfall(shap_values[0], show=False)
                    shap_fig = plt.gcf()
                    shap_fig.patch.set_facecolor(bg_col)
                    for ax_ in shap_fig.axes:
                        ax_.set_facecolor(bg_col)
                        ax_.tick_params(colors="#94a3b8")
                        for spine in ax_.spines.values():
                            spine.set_color(GRID)
                    st.pyplot(shap_fig, clear_figure=True)
                    plt.close("all")
            except Exception as e:
                    st.info(f"SHAP waterfall chart unavailable: {e}")
    # ── Tab 2: Radar ──────────────────────────────────────────────────────────
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-title">Credit Health Radar</p>', unsafe_allow_html=True)
            st.plotly_chart(radar_chart(df_input), use_container_width=True, config={"displayModeBar":False})
        with c2:
            st.markdown('<p class="section-title">Factor Breakdown</p>', unsafe_allow_html=True)
            for fname, fval in {"Payment History":float(ph),"Credit Utilization":max(0.0,100.0-cu*100),
                "Credit Age":min(100.0,ca*2.5),"Account Mix":min(100.0,na*10.0),
                "New Credit":max(0.0,100.0-hi*12.0),"Debt Ratio":max(0.0,100.0-dr*100)}.items():
                bar_color = "#34d399" if fval>=70 else "#fbbf24" if fval>=40 else "#f87171"
                st.markdown(f"""
                <div style="margin-bottom:14px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                        <span style="font-size:13px;color:#94a3b8;">{fname}</span>
                        <span style="font-size:13px;font-weight:700;color:{bar_color};">{round(fval)}/100</span>
                    </div>
                    <div style="background:{GRID};border-radius:6px;height:8px;">
                        <div style="background:{bar_color};width:{fval}%;height:8px;border-radius:6px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── Tab 3: Tips ───────────────────────────────────────────────────────────
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-title">Areas to Improve</p>', unsafe_allow_html=True)
            if suggestions:
                for feat, tip, sv_val in suggestions:
                    st.markdown(f"""<div class="tip-card"><strong>{feat.replace('_',' ')}</strong>
                    <span style="color:#f87171;font-size:12px;"> (SHAP: {sv_val:+.1f})</span><br>
                    <span style="color:#94a3b8;font-size:13px;">Tip: {tip}</span></div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="tip-card-good">All factors are positive! Great financial health.</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<p class="section-title">Your Strengths</p>', unsafe_allow_html=True)
            for feat, sv_val in sorted(positives, key=lambda x: -x[1]):
                st.markdown(f"""<div class="tip-card-good"><strong>{feat.replace('_',' ')}</strong>
                <span style="color:#34d399;font-size:12px;"> (SHAP: +{sv_val:.1f})</span></div>""", unsafe_allow_html=True)

    # ── Tab 4: AI Report ──────────────────────────────────────────────────────
    with tab4:
        if ai:
            st.markdown(f"""<div class="ai-card"><div class="ai-label">Gemini AI Credit Analysis</div>
            <p style="color:#cbd5e1;font-size:14px;line-height:1.7;margin:0;">{ai.get('summary','')}</p></div>""", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p class="section-title">Strengths</p>', unsafe_allow_html=True)
                for s in ai.get("strengths", []):
                    st.markdown(f'<div class="tip-card-good">{s}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<p class="section-title">Risk Factors</p>', unsafe_allow_html=True)
                for r in ai.get("risks", []):
                    st.markdown(f'<div class="tip-card">{r}</div>', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Recommendations</p>', unsafe_allow_html=True)
            for rec in ai.get("recommendations", []):
                impact = rec.get("impact","Medium")
                st.markdown(f"""<div class="rec-card"><span class="rec-impact impact-{impact}">{impact}</span>
                <div><strong style="color:#f1f5f9;font-size:15px;">{rec.get('title','')}</strong>
                <div style="font-size:13px;color:#94a3b8;margin-top:6px;">{rec.get('detail','')}</div>
                <div style="font-size:11px;color:#475569;margin-top:4px;">Timeline: {rec.get('timeline','')}</div></div></div>""", unsafe_allow_html=True)
            if "projection" in ai:
                st.markdown('<p class="section-title">Score Projection</p>', unsafe_allow_html=True)
                st.plotly_chart(projection_chart(score, ai["projection"]), use_container_width=True, config={"displayModeBar":False})

            st.markdown("---")
            # PDF + Email
            pdf_bytes = generate_pdf(score, band, ph, cu, ca, na, hi, dr, suggestions, positives, ai)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p class="section-title">Download PDF Report</p>', unsafe_allow_html=True)
                st.download_button("📄 Download PDF", data=pdf_bytes,
                    file_name=f"CIBIL_Report_{score}_{datetime.now().strftime('%d%m%Y')}.pdf",
                    mime="application/pdf")
            with c2:
                st.markdown('<p class="section-title">📧 Email Report</p>', unsafe_allow_html=True)
                user_email = st.text_input("Enter your email address", placeholder="you@gmail.com", key="email_input")
                if st.button("📨 Send PDF to Email", key="send_email_btn"):
                    if user_email:
                        with st.spinner("Sending email..."):
                            result = send_email_report(user_email, pdf_bytes, score, band)
                        if isinstance(result, tuple):
                            ok, msg = result
                            if ok:
                                st.success(str(msg))
                            else:
                                st.error(str(msg))
                        else:
                            st.error("Unexpected error occurred.")
                    else:
                        st.warning("Please enter your email address.")
                
        else:
            st.markdown("""<div class="ai-card" style="text-align:center;padding:40px;">
            <div style="font-size:40px;">🤖</div>
            <h3 style="color:#a78bfa;">AI Insights Unavailable</h3>
            <p style="color:#64748b;">Check your Gemini API key in the code.</p></div>""", unsafe_allow_html=True)

    # ── Tab 5: Loan Eligibility ───────────────────────────────────────────────
    with tab5:
        st.markdown('<p class="section-title">Loan Eligibility Based on Your Score</p>', unsafe_allow_html=True)
        loans = get_loan_eligibility(score)
        c1, c2 = st.columns(2)
        for i, loan in enumerate(loans):
            col = c1 if i % 2 == 0 else c2
            with col:
                st.markdown(f"""<div class="loan-card loan-{loan['status']}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div><span style="font-size:22px;">{loan['icon']}</span>
                    <strong style="color:#f1f5f9;font-size:16px;margin-left:8px;">{loan['name']}</strong></div>
                    <span style="color:{loan['color']};font-size:12px;font-weight:700;">{loan['label']}</span>
                </div>
                <div style="margin-top:10px;font-size:13px;color:#64748b;">
                    🏛️ {loan['bank']}<br>💰 {loan['amount']}<br>📊 Rate: {loan['rate']}
                </div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">Bank-wise Best Offers for You</p>', unsafe_allow_html=True)
        offers = get_bank_offers(score)
        if offers:
            for offer in offers:
                st.markdown(f"""<div class="bank-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <div><span style="font-size:20px;">{offer['logo']}</span>
                    <strong style="color:#f1f5f9;font-size:15px;margin-left:8px;">{offer['bank']}</strong>
                    <span style="color:#64748b;font-size:13px;margin-left:8px;">· {offer['product']}</span></div>
                    <span style="color:#38bdf8;font-weight:700;font-size:16px;">{offer['rate']} p.a.</span>
                </div>
                <div style="font-size:12px;color:#64748b;">
                    Processing Fee: {offer['processing']} &nbsp;|&nbsp;
                    <span style="color:#a78bfa;">⭐ {offer['special']}</span>
                </div></div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="tip-card">No bank offers available for your current score. Improve your score to unlock offers.</div>', unsafe_allow_html=True)

    # ── Tab 6: History ────────────────────────────────────────────────────────
    with tab6:
        st.markdown('<p class="section-title">Score History This Session</p>', unsafe_allow_html=True)
        if len(st.session_state.history) > 1:
            st.plotly_chart(history_chart(st.session_state.history), use_container_width=True, config={"displayModeBar":False})
        elif len(st.session_state.history) == 1:
            st.info("Predict at least 2 times to see your score trend chart.")
        st.markdown("<br>", unsafe_allow_html=True)
        for i, h in enumerate(reversed(st.session_state.history)):
            _, hcolor = score_band(h["score"])
            st.markdown(f"""<div class="history-card">
            <div><span style="font-size:11px;color:#475569;">#{len(st.session_state.history)-i} &nbsp; {h['time']}</span><br>
            <span style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;color:{hcolor};">{h['score']}</span>
            <span style="color:{hcolor};font-size:12px;margin-left:6px;">{h['band']}</span></div>
            <div style="font-size:12px;color:#475569;text-align:right;">
            PH: {h['ph']}% &nbsp; CU: {round(h['cu']*100,0)}% &nbsp; Age: {h['ca']}yr<br>
            Accts: {h['na']} &nbsp; Inq: {h['hi']} &nbsp; DTI: {round(h['dr']*100,0)}%</div></div>""", unsafe_allow_html=True)

    # ── Tab 7: What-If ────────────────────────────────────────────────────────
    with tab7:
        st.markdown('<p class="section-title">What-If Score Simulator</p>', unsafe_allow_html=True)
        st.markdown('<p style="color:#64748b;font-size:14px;">Simulate how changes affect your score without changing your main profile.</p>', unsafe_allow_html=True)
        st.markdown('<div class="whatif-card">', unsafe_allow_html=True)
        wc1, wc2 = st.columns(2)
        with wc1:
            wi_ph = st.slider("What-If Payment History (%)", 0, 100, ph, key="wi_ph")
            wi_cu = st.slider("What-If Credit Utilization", 0.0, 1.0, cu, step=0.01, key="wi_cu")
            wi_ca = st.slider("What-If Credit Age (Years)", 0, 50, ca, key="wi_ca")
        with wc2:
            wi_na = st.slider("What-If Number of Accounts", 1, 100, na, key="wi_na")
            wi_hi = st.slider("What-If Hard Inquiries", 0, 10, hi, key="wi_hi")
            wi_dr = st.slider("What-If Debt-to-Income", 0.0, 1.0, dr, step=0.01, key="wi_dr")
        st.markdown('</div>', unsafe_allow_html=True)
        wi_score, _, _ = predict_cibil(model, explainer, [wi_ph, wi_cu, wi_ca, wi_na, wi_hi, wi_dr])
        wi_band, wi_color = score_band(wi_score)
        score_diff = wi_score - score
        diff_color = "#34d399" if score_diff>0 else "#f87171" if score_diff<0 else "#64748b"
        diff_sign  = "+" if score_diff>0 else ""
        wc1, wc2, wc3 = st.columns(3)
        for col, lbl, val, col_val, bnd in [(wc1,"CURRENT SCORE",score,color,band),(wc2,"WHAT-IF SCORE",wi_score,wi_color,wi_band),
            (wc3,"DIFFERENCE",f"{diff_sign}{score_diff}",diff_color,"Improvement" if score_diff>0 else "Decrease" if score_diff<0 else "No Change")]:
            with col:
                st.markdown(f"""<div class="metric-chip" style="padding:20px;">
                <div style="font-size:11px;color:#64748b;letter-spacing:1px;">{lbl}</div>
                <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;color:{col_val};">{val}</div>
                <div style="font-size:12px;color:{col_val};">{bnd}</div></div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(whatif_chart(score, wi_score), use_container_width=True, config={"displayModeBar":False})

    # ── Tab 8: EMI Calculator ─────────────────────────────────────────────────
    with tab8:
        st.markdown('<p class="section-title">EMI Calculator</p>', unsafe_allow_html=True)
        ec1, ec2 = st.columns(2)
        with ec1:
            loan_type = st.selectbox("Loan Type", ["Home Loan","Car Loan","Personal Loan","Education Loan","Business Loan"])
            principal = st.number_input("Loan Amount (₹)", min_value=10000, max_value=50000000, value=1000000, step=10000)
            rate_map  = {"Home Loan":8.5,"Car Loan":9.5,"Personal Loan":12.0,"Education Loan":9.0,"Business Loan":13.0}
            annual_rate = st.slider("Annual Interest Rate (%)", 5.0, 25.0, rate_map.get(loan_type, 10.0), step=0.1)
            tenure_yr   = st.slider("Loan Tenure (Years)", 1, 30, 10)
            tenure_months = tenure_yr * 12

            emi, total_payment, total_interest = calculate_emi(principal, annual_rate, tenure_months)

            st.markdown(f"""<div class="emi-result">
            <div style="font-size:12px;letter-spacing:2px;color:#64748b;margin-bottom:8px;">MONTHLY EMI</div>
            <div style="font-family:'Syne',sans-serif;font-size:56px;font-weight:800;color:#38bdf8;">
                ₹{emi:,.0f}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:20px;">
                <div><div style="font-size:11px;color:#64748b;">Total Payment</div>
                <div style="font-size:18px;font-weight:700;color:#f1f5f9;">₹{total_payment:,.0f}</div></div>
                <div><div style="font-size:11px;color:#64748b;">Total Interest</div>
                <div style="font-size:18px;font-weight:700;color:#f87171;">₹{total_interest:,.0f}</div></div>
            </div></div>""", unsafe_allow_html=True)

            # Score-based rate advice
            _, rate_color = score_band(score)
            if score >= 750:
                advice = f"With your score of {score}, you qualify for the BEST rates. Negotiate hard!"
            elif score >= 700:
                advice = f"With your score of {score}, you get standard rates. Improve to 750+ for better deals."
            else:
                advice = f"With your score of {score}, rates will be higher. Improve your score first."
            st.markdown(f"""<div class="tip-card-good" style="margin-top:14px;">
            <span style="color:{rate_color};">💡 {advice}</span></div>""", unsafe_allow_html=True)

        with ec2:
            st.markdown('<p class="section-title">Payment Breakdown</p>', unsafe_allow_html=True)
            st.plotly_chart(emi_breakdown_chart(principal, total_interest), use_container_width=True, config={"displayModeBar":False})

            # Amortization table
            st.markdown('<p class="section-title">Amortization Schedule (First 12 Months)</p>', unsafe_allow_html=True)
            r = annual_rate / (12 * 100)
            bal = principal
            amort = []
            for month in range(1, min(13, tenure_months+1)):
                interest_part = bal * r
                principal_part = emi - interest_part
                bal -= principal_part
                amort.append({"Month": month, "EMI": f"₹{emi:,.0f}",
                    "Principal": f"₹{principal_part:,.0f}", "Interest": f"₹{interest_part:,.0f}",
                    "Balance": f"₹{max(0,bal):,.0f}"})
            st.dataframe(pd.DataFrame(amort), use_container_width=True, hide_index=True)

    # ── Tab 9: Roadmap ────────────────────────────────────────────────────────
    with tab9:
        st.markdown('<p class="section-title">Your Personalized Score Improvement Roadmap</p>', unsafe_allow_html=True)
        roadmap_steps, target = get_roadmap(score, suggestions)

        _, target_color = score_band(target)
        st.markdown(f"""<div class="whatif-card" style="text-align:center;margin-bottom:24px;">
        <div style="font-size:13px;color:#64748b;letter-spacing:1px;">CURRENT SCORE → TARGET SCORE</div>
        <div style="display:flex;justify-content:center;align-items:center;gap:24px;margin-top:10px;">
            <div style="font-family:'Syne',sans-serif;font-size:48px;font-weight:800;color:{color};">{score}</div>
            <div style="font-size:28px;color:#64748b;">→</div>
            <div style="font-family:'Syne',sans-serif;font-size:48px;font-weight:800;color:{target_color};">{target}+</div>
        </div></div>""", unsafe_allow_html=True)

        for step in roadmap_steps:
            st.markdown(f"""<div class="roadmap-step">
            <div style="background:{step['color']}22;border:2px solid {step['color']};border-radius:50%;
                width:44px;height:44px;display:flex;align-items:center;justify-content:center;
                font-size:20px;flex-shrink:0;">{step['icon']}</div>
            <div style="flex:1;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <strong style="color:#f1f5f9;font-size:15px;">Step {step['step']}: {step['title']}</strong>
                    <span style="color:{step['color']};font-size:12px;font-weight:700;">{step['impact']}</span>
                </div>
                <div style="font-size:13px;color:#94a3b8;margin-top:4px;">{step['detail']}</div>
                <div style="font-size:11px;color:#475569;margin-top:4px;">⏱ {step['timeline']}</div>
            </div></div>""", unsafe_allow_html=True)

    # ── Tab 10: Compare ───────────────────────────────────────────────────────
    with tab10:
        st.markdown('<p class="section-title">Your Score vs Average Indian CIBIL Score</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(compare_chart(score), use_container_width=True, config={"displayModeBar":False})
        with c2:
            avg = 720; diff_avg = score - avg
            diff_c = "#34d399" if diff_avg >= 0 else "#f87171"
            diff_s = "+" if diff_avg >= 0 else ""
            st.markdown(f"""
            <div style="margin-top:20px;">
                <div class="metric-chip" style="margin-bottom:12px;padding:20px;">
                    <div style="font-size:11px;color:#64748b;">VS AVERAGE INDIAN (720)</div>
                    <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;color:{diff_c};">{diff_s}{diff_avg}</div>
                    <div style="font-size:12px;color:{diff_c};">{"Above" if diff_avg>=0 else "Below"} Average</div>
                </div>
                <div class="metric-chip" style="padding:20px;">
                    <div style="font-size:11px;color:#64748b;">PERCENTILE</div>
                    <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;color:{color};">{percentile}th</div>
                    <div style="font-size:12px;color:{color};">Better than {percentile}% of Indians</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">Score Band Distribution in India</p>', unsafe_allow_html=True)
        band_data = [
            {"Band":"Poor (300-579)",   "Population":"~15%","color":"#f87171"},
            {"Band":"Fair (580-669)",   "Population":"~20%","color":"#fbbf24"},
            {"Band":"Good (670-739)",   "Population":"~25%","color":"#a3e635"},
            {"Band":"Very Good (740-799)","Population":"~25%","color":"#34d399"},
            {"Band":"Exceptional (800+)","Population":"~15%","color":"#38bdf8"},
        ]
        for bd in band_data:
            is_yours = band.lower() in bd["Band"].lower()
            border = f"border:2px solid {bd['color']};" if is_yours else ""
            you_tag = f'<span style="color:{bd["color"]};font-weight:700;margin-left:8px;">← You are here</span>' if is_yours else ""
            st.markdown(f"""<div class="compare-bar" style="{border}">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="color:{bd['color']};font-weight:600;">{bd['Band']}{you_tag}</span>
                <span style="color:#64748b;font-size:13px;">{bd['Population']} of Indians</span>
            </div></div>""", unsafe_allow_html=True)

else:
    welcome_text = t('welcome')
    predict_text = t('predict_btn')
    st.markdown(f"""
    <div class="welcome-box">
        <div style="font-size:48px;">💳</div>
        <h2 style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
                   background:linear-gradient(to right,#38bdf8,#a78bfa);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            {welcome_text}
        </h2>
        <p style="color:#64748b;font-size:15px;max-width:560px;margin:14px auto 0;">
            Adjust your financial profile in the sidebar and click
            <strong style="color:#38bdf8;">{predict_text}</strong>
        </p>
        <div style="display:flex;justify-content:center;gap:20px;margin-top:36px;flex-wrap:wrap;">
            <div style="text-align:center;"><div style="font-size:26px;">📊</div><div style="color:#475569;font-size:11px;margin-top:4px;">SHAP</div></div>
            <div style="text-align:center;"><div style="font-size:26px;">🤖</div><div style="color:#475569;font-size:11px;margin-top:4px;">Gemini AI</div></div>
            <div style="text-align:center;"><div style="font-size:26px;">🏦</div><div style="color:#475569;font-size:11px;margin-top:4px;">Loan Eligibility</div></div>
            <div style="text-align:center;"><div style="font-size:26px;">🧮</div><div style="color:#475569;font-size:11px;margin-top:4px;">EMI Calculator</div></div>
            <div style="text-align:center;"><div style="font-size:26px;">🗺️</div><div style="color:#475569;font-size:11px;margin-top:4px;">Roadmap</div></div>
            <div style="text-align:center;"><div style="font-size:26px;">📈</div><div style="color:#475569;font-size:11px;margin-top:4px;">History</div></div>
            <div style="text-align:center;"><div style="font-size:26px;">🔮</div><div style="color:#475569;font-size:11px;margin-top:4px;">What-If</div></div>
            <div style="text-align:center;"><div style="font-size:26px;">📄</div><div style="color:#475569;font-size:11px;margin-top:4px;">PDF + Email</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    AutoCred &nbsp;·&nbsp; XGBoost + SHAP + Gemini AI &nbsp;·&nbsp; For educational purposes only
</div>""", unsafe_allow_html=True)