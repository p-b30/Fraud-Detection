# app.py
import os
import json
import joblib
import requests
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

import plotly.express as px
import plotly.graph_objects as go

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Lottie helper
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Fraud Detection Intelligence System",
                   layout="wide",
                   initial_sidebar_state="expanded")

# --------------------
# CSS + Lottie background
# --------------------
LOTTIE_BG_URL = "https://assets1.lottiefiles.com/packages/lf20_tfb3estd.json"  # example public bg

def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

lottie_bg_json = load_lottieurl(LOTTIE_BG_URL)

st.markdown(
    """
    <style>
    /* Remove default padding and set background overlay */
    .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .lottie-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -1;
        opacity: 0.18;
        pointer-events: none;
    }
    .overlay {
        position: fixed;
        inset: 0;
        background: rgba(5,10,20,0.45);
        z-index: -0.5;
        pointer-events: none;
    }
    /* uploader label color */
    div[data-testid="stFileUploader"] label {
        color: #FFDD00 !important;
        font-weight: bold;
        font-size: 15px;
    }
    /* small card look */
    .metric-card {
        background: rgba(255,255,255,0.04);
        padding: 12px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if lottie_bg_json:
    st.markdown('<div class="lottie-bg">', unsafe_allow_html=True)
    st_lottie(lottie_bg_json, height=900, key="bg_lottie", quality="low", loop=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)

# --------------------
# Utility functions
# --------------------
DATA_PATH = Path("data/creditcard.csv")
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_FILE = MODEL_PATH / "fraud_model.pkl"
SCALER_FILE = MODEL_PATH / "scaler.pkl"

@st.cache_data
def load_local_data(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)

def preprocess(df, scaler=None, fit_scaler=False):
    dfc = df.copy()
    # If dataset has Time, Amount and PCA features (common Kaggle creditcard)
    if "Class" not in dfc.columns:
        raise ValueError("Expected column 'Class' for labels.")
    X = dfc.drop(columns=["Class"])
    y = dfc["Class"]

    # Scale Amount and Time if present
    cols_to_scale = []
    if "Amount" in X.columns:
        cols_to_scale.append("Amount")
    if "Time" in X.columns:
        cols_to_scale.append("Time")

    if cols_to_scale:
        if scaler is None and fit_scaler:
            scaler = StandardScaler()
            X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale].values)
        elif scaler is not None:
            X[cols_to_scale] = scaler.transform(X[cols_to_scale].values)

    return X, y, scaler

def encode_categoricals(df):
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
    return df_encoded


def train_model(df, undersample=True, n_estimators=120):
    df = encode_categoricals(df)
    X, y, scaler = preprocess(df, scaler=None, fit_scaler=True)

    # Handle imbalance quickly with undersampling (fast)
    if undersample:
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
    else:
        X_res, y_res = X, y

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Save scaler (fit earlier)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(model, MODEL_FILE)

    # Evaluate on original test split (sample from original dataset)
    X_full, y_full, _ = preprocess(df, scaler=scaler, fit_scaler=False)
    _, X_hold, _, y_hold = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
    y_pred = model.predict(X_hold)
    y_proba = model.predict_proba(X_hold)[:,1]

    report = classification_report(y_hold, y_pred, output_dict=True)
    roc = roc_auc_score(y_hold, y_proba)
    return model, scaler, report, roc

def load_model_if_exists():
    if MODEL_FILE.exists() and SCALER_FILE.exists():
        try:
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            return model, scaler
        except Exception:
            return None, None
    return None, None

def predict_df(model, scaler, df):
    X = df.copy()
    if "Class" in X.columns:
        X = X.drop(columns=["Class"])
    # scale Time & Amount if present
    cols_to_scale = [c for c in ("Time","Amount") if c in X.columns]
    if cols_to_scale and scaler:
        X[cols_to_scale] = scaler.transform(X[cols_to_scale])
    probs = model.predict_proba(X)[:,1]
    preds = (probs >= 0.5).astype(int)
    out = X.copy()
    out["fraud_probability"] = probs
    out["prediction"] = preds
    return out

# --------------------
# UI Layout
# --------------------
st.title("💳 Fraud Detection Intelligence System")
st.write("Interactive dashboard with ML model & explainability. Upload transactions or use the sample dataset to train & test.")

col1, col2 = st.columns([2,1])

with col2:
    st.markdown("## Controls")
    uploaded = st.file_uploader("Upload CSV with transactions (columns like V1..V28, Amount, Time, Class optional)", type=["csv"])
    use_sample = st.button("Use local sample dataset (data/creditcard.csv)")
    retrain = st.button("Retrain model (may take a minute)")
    undersample = st.checkbox("Use undersampling while training (fast, recommended)", value=True)
    explain_toggle = st.checkbox("Enable SHAP explainability (if installed)", value=False)
    st.markdown("---")
    st.markdown("### Model status")
    model, scaler = load_model_if_exists()
    if model:
        st.success("Model loaded from models/fraud_model.pkl")
    else:
        st.info("No trained model found. Train model by clicking 'Retrain model' or upload a pre-trained model file.")

# --------------------
# Load data
# --------------------
df = None
# --- File upload handling ---
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success("✅ Uploaded CSV loaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to read uploaded file: {e}")

# --- Local sample button ---
elif use_sample:
    df = load_local_data(DATA_PATH)
    if df is None:
        st.warning("⚠️ Sample dataset not found at data/creditcard.csv. Using built-in demo dataset instead.")
        df = pd.DataFrame({
            'TransactionID': [1, 2, 3, 4, 5],
            'Amount': [120.5, 2300.0, 45.0, 678.9, 90.5],
            'TransactionType': ['Online', 'In-Store', 'Online', 'Online', 'In-Store'],
            'AccountAgeDays': [300, 1200, 45, 780, 230],
            'FraudFlag': [0, 1, 0, 1, 0]
        })
    else:
        st.success("✅ Local sample dataset loaded.")

# --- Default fallback demo dataset ---
elif uploaded is None and not use_sample:
    st.info("No file uploaded — showing demo fraud dataset.")
    df = pd.DataFrame({
        'TransactionID': [1, 2, 3, 4, 5],
        'Amount': [120.5, 2300.0, 45.0, 678.9, 90.5],
        'TransactionType': ['Online', 'In-Store', 'Online', 'Online', 'In-Store'],
        'AccountAgeDays': [300, 1200, 45, 780, 230],
        'FraudFlag': [0, 1, 0, 1, 0]
    })

# --------------------
# ✅ Normalize target column name
# --------------------
if df is not None:
    # Rename 'IsFraud' or 'FraudFlag' to 'Class' (the label column)
    if 'IsFraud' in df.columns:
        df.rename(columns={'IsFraud': 'Class'}, inplace=True)
    elif 'FraudFlag' in df.columns:
        df.rename(columns={'FraudFlag': 'Class'}, inplace=True)

    st.success("✅ Dataset ready with column 'Class' as label.")
else:
    st.error("❌ No dataset found or loaded.")

# --------------------
# Retrain logic
# --------------------
if retrain:
    # require local sample or uploaded dataset
    if df is None:
        st.error("No dataset loaded. Upload CSV or use local sample dataset to train.")
    else:
        with st.spinner("Training model — this may take 30s+ depending on dataset and machine..."):
            try:
                model, scaler, report, roc = train_model(df, undersample=undersample, n_estimators=120)
                st.success("Model trained & saved to models/fraud_model.pkl")
                st.write("ROC-AUC on holdout:", round(roc,4))
                st.json(report)
            except Exception as e:
                st.error(f"Training failed: {e}")

# If we have df loaded, show insights
if df is not None:
    st.markdown("## Data Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", df.shape[0])
    with c2:
        st.metric("Columns", df.shape[1])
    with c3:
        fraud_count = int(df['Class'].sum()) if 'Class' in df.columns else "N/A"
        st.metric("Fraud Count", fraud_count)

    st.markdown("### Quick distribution")
    if "Class" in df.columns:
        fig = px.histogram(df, x="Class", title="Fraud(1) vs Non-Fraud(0) distribution", labels={"Class":"Class"})
        st.plotly_chart(fig, use_container_width=True)
    if "Amount" in df.columns:
        fig2 = px.histogram(df, x="Amount", nbins=80, title="Transaction Amount Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    # Correlation heatmap (safe for mixed data)
st.markdown("### Correlation heatmap (numeric columns only)")
try:
    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        sample_corr = numeric_df.sample(min(len(numeric_df), 2000), random_state=42).corr()
        fig3 = px.imshow(sample_corr, title="Correlation matrix (numeric only, up to 2000 rows)")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No numeric columns found for correlation.")
except Exception as e:
    st.warning(f"Could not generate correlation heatmap: {e}")

    # Show top rows
    st.markdown("### Data preview")
    st.dataframe(df.head(200))

# --------------------




# --- Retrain Model Section ---
st.subheader("🧠 Retrain Model")

if st.button("Train Model using Sample Data"):
    if df is not None:
        df_encoded = df.copy()

        # Convert categorical columns to numeric
        for col in df_encoded.select_dtypes(include=['object']).columns:
            df_encoded[col] = df_encoded[col].astype('category').cat.codes

        # --- Identify the target column ---
        target_col = None
        if 'Class' in df_encoded.columns:
            target_col = 'Class'
        elif 'IsFraud' in df_encoded.columns:
            target_col = 'IsFraud'
        elif 'FraudFlag' in df_encoded.columns:
            target_col = 'FraudFlag'

        if target_col:
            X = df_encoded.drop(target_col, axis=1)
            y = df_encoded[target_col]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Save model
            import os
            os.makedirs("models", exist_ok=True)
            with open("models/fraud_model.pkl", "wb") as f:
                pickle.dump(model, f)

            st.success(f"✅ Model trained successfully using '{target_col}' as label and saved as 'models/fraud_model.pkl'. You can now predict transactions!")
        else:
            st.error("❌ No valid target column found. Expected one of: 'Class', 'IsFraud', or 'FraudFlag'.")
    else:
        st.error("⚠️ Please load or use the demo dataset first.")


# --------------------
# 🧠 Auto-train model if missing
# --------------------
if model is None:
    st.warning("No trained model found — training a lightweight model automatically...")
    try:
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include=['object']).columns:
            df_encoded[col] = df_encoded[col].astype('category').cat.codes

        # Detect label column
        target_col = None
        for col in ['Class', 'IsFraud', 'FraudFlag']:
            if col in df_encoded.columns:
                target_col = col
                break

        if target_col:
            X = df_encoded.drop(columns=[target_col])
            y = df_encoded[target_col]
            test_size = 0.3 if len(df_encoded) > 10 else 0.5
            X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(y.unique()) > 1 else None
            )
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            os.makedirs("models", exist_ok=True)
            with open("models/fraud_model.pkl", "wb") as f:
                pickle.dump(model, f)
            st.success("✅ Auto-trained model saved as models/fraud_model.pkl.")
        else:
            st.error("❌ No valid label column found (expected 'Class', 'IsFraud', or 'FraudFlag').")
    except Exception as e:
        st.error(f"⚠️ Auto-training failed: {e}")




# Prediction on uploaded transactions or sample
# --------------------
st.markdown("---")
st.markdown("## Predict transactions")

if model is None:
    st.info("No trained model loaded. Upload a model `models/fraud_model.pkl` into models/ or retrain the model.")
else:
    # If user uploaded a CSV and it doesn't contain Class -> run predictions
    predict_df_ui = None
    if uploaded is not None:
        try:
            to_predict = pd.read_csv(uploaded)
            predict_df_ui = to_predict
        except Exception:
            predict_df_ui = None
    elif df is not None:
        # let user choose to predict on the sample
        if st.button("Run predictions on the loaded dataset"):
            predict_df_ui = df.copy()

    if predict_df_ui is not None:
        with st.spinner("Running predictions..."):
            try:
                results = predict_df(model, scaler, predict_df_ui)
                st.success("Predictions complete.")
                # show top suspicious transactions
                st.markdown("### Top risky transactions")
                top = results.sort_values("fraud_probability", ascending=False).head(20)
                st.dataframe(top.style.format({"fraud_probability":"{:.4f}"}))
                # plot risk distribution
                figp = px.histogram(results, x="fraud_probability", nbins=50, title="Fraud probability distribution")
                st.plotly_chart(figp, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Single-transaction prediction UI
    st.markdown("### Predict single transaction (enter features present in your dataset)")
    st.info("Enter only features that exist in your model dataset (e.g., Amount, Time, V1..V28). Leave others blank.")
    with st.form("single_tx_form"):
        # We'll try to auto-populate fields from scaler/model if possible
        sample_fields = []
        # attempt to get columns used by scaler/model from sample dataset if available
        if df is not None:
            sample_fields = [c for c in df.columns if c != "Class"]
        # show a few common ones
        amt = st.text_input("Amount", value="0")
        tim = st.text_input("Time", value="0")
        submitted = st.form_submit_button("Predict")

    if submitted:
        # build a DataFrame with available fields
        # Minimal: Amount & Time if model expects them
        tx = {}
        if df is not None:
            # create zero row for all features except Class
            zero_row = {c:0 for c in df.columns if c!="Class"}
            # set entered amount/time
            if "Amount" in zero_row:
                zero_row["Amount"] = float(amt)
            if "Time" in zero_row:
                zero_row["Time"] = float(tim)
            tx = pd.DataFrame([zero_row])
        else:
            tx = pd.DataFrame([{"Amount": float(amt), "Time": float(tim)}])

        try:
            out = predict_df(model, scaler, tx)
            prob = out["fraud_probability"].iloc[0]
            pred = out["prediction"].iloc[0]
            st.metric("Fraud probability", f"{prob:.3f}")
            st.metric("Prediction", "FRAUD" if pred==1 else "NOT FRAUD")
            # SHAP explanation for single transaction
            if SHAP_AVAILABLE and explain_toggle:
                st.markdown("#### SHAP explanation")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(tx)
                shap.initjs()
                st_shap = st.pyplot
                try:
                    shap.force_plot(explainer.expected_value[1], shap_values[1], tx, matplotlib=True, show=True)
                    st.pyplot(bbox_inches="tight")
                except Exception:
                    st.warning("SHAP force_plot failed for display. Try installing the latest shap and run locally.")
            elif explain_toggle and not SHAP_AVAILABLE:
                st.warning("SHAP is not installed. Install with: python -m pip install shap")
        except Exception as e:
            st.error(f"Failed to predict single transaction: {e}")

# --------------------
# Extra: Show model performance if available
# --------------------
if model is not None and df is not None:
    st.markdown("---")
    st.markdown("## Model performance (quick eval)")
    try:
        # 🔹 Encode categorical columns before evaluation
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include=['object']).columns:
            df_encoded[col] = df_encoded[col].astype('category').cat.codes

        # evaluate on a holdout subset from original DF
        X_full, y_full, _ = preprocess(df_encoded, scaler=scaler, fit_scaler=False)

# dynamically choose test size to avoid errors on small datasets
        if len(df_encoded) < 10:
            test_size = 0.5
        else:
            test_size = 0.3

        stratify_labels = y_full if len(y_full.unique()) > 1 else None
        X_train, X_hold, y_train, y_hold = train_test_split(
            X_full, y_full, test_size=test_size, random_state=42, stratify=stratify_labels
        )

        
        y_pred = model.predict(X_hold)
        y_proba = model.predict_proba(X_hold)[:,1]
        auc = roc_auc_score(y_hold, y_proba)
        report_text = classification_report(y_hold, y_pred, output_dict=False)
        st.markdown(f"**ROC-AUC (holdout):** {auc:.4f}")
        st.text(report_text)
        cm = confusion_matrix(y_hold, y_pred)
        figcm = go.Figure(data=go.Heatmap(z=cm, x=["Pred Non-Fraud","Pred Fraud"], y=["True Non-Fraud","True Fraud"],
                                         colorscale="Viridis"))
        figcm.update_layout(title="Confusion Matrix")
        st.plotly_chart(figcm, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to evaluate model: {e}")

try:
    sample_data = pd.read_csv("fraud_data.csv")
    st.markdown("### 📊 Local Sample Fraud Data")
    st.dataframe(sample_data)
except FileNotFoundError:
    st.warning("fraud_data.csv not found in your project folder.")



st.markdown("---")
st.caption("Built with ❤️ — Fraud Detection Intelligence System. Save your trained model in models/fraud_model.pkl for reuse.")
