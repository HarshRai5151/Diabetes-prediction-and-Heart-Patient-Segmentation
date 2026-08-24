import streamlit as st
import numpy as np
import joblib
import time
import os

# ──────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediAI Diagnostics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. CUSTOM CSS — polished look & feel
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main header gradient */
    .main-header {
        background: linear-gradient(135deg, #0d6efd 0%, #6610f2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; }
    .main-header p  { color: #ffffffcc; margin: 0.4rem 0 0 0; }

    /* Result cards */
    .result-card {
        padding: 1.8rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .result-healthy  { background: linear-gradient(135deg, #d4edda, #c3e6cb); border-left: 6px solid #28a745; }
    .result-diabetic { background: linear-gradient(135deg, #f8d7da, #f5c6cb); border-left: 6px solid #dc3545; }

    .cluster-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-top: 1rem;
    }
    .cluster-low      { background: linear-gradient(135deg, #d4edda, #c3e6cb); border-left: 6px solid #28a745; }
    .cluster-moderate  { background: linear-gradient(135deg, #fff3cd, #ffeeba); border-left: 6px solid #ffc107; }
    .cluster-high      { background: linear-gradient(135deg, #f8d7da, #f5c6cb); border-left: 6px solid #dc3545; }

    /* Sidebar polish */
    section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 3. HELPER — safe model loader (cached)
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


@st.cache_resource
def load_pickle(filename: str):
    """Load a single .pkl file; returns None on failure."""
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.isfile(path):
        return None
    return joblib.load(path)


# ──────────────────────────────────────────────────────────────────────────────
# 4. SIDEBAR — navigation & about
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MediAI Diagnostics")
    st.caption("AI-Powered Clinical Decision Support")
    st.markdown("---")

    page = st.radio(
        "**Navigate**",
        ["🩸 Diabetes Prediction", "❤️ Heart Patient Segmentation"],
        index=0,
    )

    st.markdown("---")

    with st.expander("ℹ️ About This App", expanded=False):
        st.markdown(
            """
            This dashboard assists healthcare professionals with
            **early-stage diagnostic screening** powered by Machine Learning.

            | Module | Technique |
            |--------|-----------|
            | Diabetes | Random Forest Classifier |
            | Heart Segmentation | StandardScaler → PCA → K-Means |

            > **Disclaimer:** This tool is for **educational / research purposes
            > only** and must not replace professional medical advice.
            """
        )

    st.markdown(
        "<p style='text-align:center;color:grey;font-size:0.75rem;'>"
        "© 2026 MediAI Diagnostics</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DIABETES RISK PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "🩸 Diabetes Prediction":

    # Header
    st.markdown(
        '<div class="main-header">'
        "<h1>🩸 Diabetes Risk Assessment</h1>"
        "<p>Enter patient clinical data below to estimate the probability of diabetes using a Random Forest model.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Load model
    diabetes_model = load_pickle("diabetes_model.pkl")
    if diabetes_model is None:
        st.warning("⚠️ **Model file not found:** `models/diabetes_model.pkl` is missing. Please ensure the file exists and restart the app.")
        st.stop()

    # Model explainer
    with st.expander("📖 How does this model work?"):
        st.markdown(
            """
            **Algorithm:** Random Forest Classifier

            The model was trained on the **Pima Indians Diabetes Dataset** with 8 clinical
            features.  It builds an ensemble of decision trees and aggregates their votes to
            produce both a **binary prediction** (Diabetic / Healthy) and a **confidence
            probability**.

            | Feature | Description |
            |---------|-------------|
            | Pregnancies | Number of times pregnant |
            | Glucose | Plasma glucose concentration (mg/dL) |
            | Blood Pressure | Diastolic blood pressure (mm Hg) |
            | Skin Thickness | Triceps skinfold thickness (mm) |
            | Insulin | 2-hour serum insulin (µU/mL) |
            | BMI | Body mass index (kg/m²) |
            | Diabetes Pedigree Function | Genetic risk score |
            | Age | Age in years |
            """
        )

    # ── Input Form ──────────────────────────────────────────────────────────
    st.subheader("📝 Patient Information")

    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input(
                "Pregnancies", min_value=0, max_value=20, value=1, step=1,
                help="Number of times the patient has been pregnant",
            )
            glucose = st.slider(
                "Glucose Level (mg/dL)", min_value=0, max_value=300, value=120,
                help="Plasma glucose concentration after 2-hour oral glucose tolerance test",
            )
            blood_pressure = st.slider(
                "Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70,
                help="Diastolic blood pressure",
            )
            skin_thickness = st.number_input(
                "Skin Thickness (mm)", min_value=0, max_value=100, value=20,
                help="Triceps skinfold thickness",
            )

        with col2:
            insulin = st.number_input(
                "Insulin (µU/mL)", min_value=0, max_value=900, value=79,
                help="2-hour serum insulin level",
            )
            bmi = st.slider(
                "BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1,
                help="Body mass index",
            )
            dpf = st.number_input(
                "Diabetes Pedigree Function", min_value=0.000, max_value=3.000,
                value=0.500, step=0.001, format="%.3f",
                help="Genetic diabetes risk function score",
            )
            age = st.slider(
                "Age (years)", min_value=1, max_value=120, value=33,
            )

        submitted = st.form_submit_button("🔬  Run Diagnosis", use_container_width=True)

    # ── Prediction ──────────────────────────────────────────────────────────
    if submitted:
        input_array = np.array(
            [[pregnancies, glucose, blood_pressure, skin_thickness,
              insulin, bmi, dpf, age]]
        )

        with st.spinner("Analyzing patient vitals …"):
            time.sleep(0.6)
            try:
                prediction = diabetes_model.predict(input_array)[0]
                probabilities = diabetes_model.predict_proba(input_array)[0]
                prob_positive = probabilities[1]
            except Exception as exc:
                st.error(f"Prediction failed — please verify input values.  \nDetails: `{exc}`")
                st.stop()

        st.markdown("---")
        st.subheader("🔍 Diagnostic Results")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if prediction == 1:
                st.markdown(
                    '<div class="result-card result-diabetic">'
                    '<h2 style="color:#dc3545;">🚨 HIGH RISK</h2>'
                    f'<h3 style="color:#721c24;">{prob_positive:.1%} Confidence</h3>'
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-card result-healthy">'
                    '<h2 style="color:#28a745;">✅ LOW RISK</h2>'
                    f'<h3 style="color:#155724;">{1 - prob_positive:.1%} Confidence</h3>'
                    "</div>",
                    unsafe_allow_html=True,
                )

        with res_col2:
            st.markdown("**Probability Breakdown**")
            prob_col_a, prob_col_b = st.columns(2)
            with prob_col_a:
                st.metric("Diabetic Probability", f"{prob_positive:.1%}")
                st.progress(float(prob_positive))
            with prob_col_b:
                st.metric("Healthy Probability", f"{1 - prob_positive:.1%}")
                st.progress(float(1 - prob_positive))

            st.markdown("")
            if prediction == 1:
                st.warning(
                    "⚠️ **Recommendation:** Schedule a follow-up HbA1c test and "
                    "consult an endocrinologist at the earliest convenience."
                )
            else:
                st.info(
                    "ℹ️ **Recommendation:** Continue maintaining a healthy lifestyle. "
                    "No immediate clinical action required."
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — HEART PATIENT SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "❤️ Heart Patient Segmentation":

    # Header
    st.markdown(
        '<div class="main-header">'
        "<h1>❤️ Heart Disease Patient Segmentation</h1>"
        "<p>Classify patients into clinical risk groups using Unsupervised Learning (K-Means Clustering).</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Load models
    scaler = load_pickle("scaler.pkl")
    pca = load_pickle("pca.pkl")
    kmeans = load_pickle("kmeans_model.pkl")

    missing = []
    if scaler is None:
        missing.append("`models/scaler.pkl`")
    if pca is None:
        missing.append("`models/pca.pkl`")
    if kmeans is None:
        missing.append("`models/kmeans_model.pkl`")

    if missing:
        st.warning(
            f"⚠️ **Missing model file(s):** {', '.join(missing)}.  \n"
            "Please ensure all files exist in the `models/` directory and restart the app."
        )
        st.stop()

    # Model explainer
    with st.expander("📖 How does this model work?"):
        st.markdown(
            """
            **Pipeline:** StandardScaler → Feature Selection → K-Means

            1. **Scaling** — Five numeric features (age, trestbps, chol, thalach, oldpeak)
               are standardised to zero mean and unit variance.
            2. **One-Hot Encoding** — Categorical features (cp, slope, thal) are converted
               to binary dummy variables.
            3. **Feature Selection** — The top 10 most discriminative features (selected
               via ANOVA F-test during training) are used for clustering.
            4. **K-Means Clustering** — Patients are grouped into 3 distinct clusters based
               on their cardiovascular profile.

            The model was trained on the **UCI Heart Disease** dataset.
            """
        )

    # ── Input Form ──────────────────────────────────────────────────────────
    st.subheader("📝 Patient Cardiovascular Profile")
    st.caption("Standard UCI Heart Disease feature set")

    with st.form("heart_form"):
        col1, col2 = st.columns(2)

        with col1:
            age_h = st.slider("Age", min_value=20, max_value=100, value=50)
            sex_h = st.selectbox(
                "Sex", options=[0, 1],
                format_func=lambda x: "👩 Female" if x == 0 else "👨 Male",
            )
            cp_h = st.selectbox(
                "Chest Pain Type",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0 — Typical Angina",
                    1: "1 — Atypical Angina",
                    2: "2 — Non-anginal Pain",
                    3: "3 — Asymptomatic",
                }[x],
            )
            trestbps_h = st.slider(
                "Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120,
            )
            chol_h = st.number_input(
                "Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200,
            )
            fbs_h = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            restecg_h = st.selectbox(
                "Resting ECG Results",
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "0 — Normal",
                    1: "1 — ST-T Wave Abnormality",
                    2: "2 — Left Ventricular Hypertrophy",
                }[x],
            )

        with col2:
            thalach_h = st.slider(
                "Max Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=150,
            )
            exang_h = st.selectbox(
                "Exercise-Induced Angina",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            oldpeak_h = st.number_input(
                "ST Depression (oldpeak)", min_value=0.0, max_value=10.0,
                value=1.0, step=0.1,
                help="ST depression induced by exercise relative to rest",
            )
            slope_h = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "0 — Upsloping",
                    1: "1 — Flat",
                    2: "2 — Downsloping",
                }[x],
            )
            ca_h = st.selectbox(
                "Number of Major Vessels (0-4)", options=[0, 1, 2, 3, 4],
                help="Number of major vessels coloured by fluoroscopy",
            )
            thal_h = st.selectbox(
                "Thalassemia",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0 — Normal",
                    1: "1 — Fixed Defect",
                    2: "2 — Reversible Defect",
                    3: "3 — Other",
                }[x],
            )

        segment_btn = st.form_submit_button(
            "🫀  Identify Patient Segment", use_container_width=True,
        )

    # ── Prediction ──────────────────────────────────────────────────────────
    if segment_btn:
        with st.spinner("Processing cardiovascular profile …"):
            time.sleep(0.6)
            try:
                # Step 1 — Scale only the 5 numeric features the scaler was trained on
                import pandas as pd

                numeric_raw = np.array([[age_h, trestbps_h, chol_h, thalach_h, oldpeak_h]])
                numeric_scaled = scaler.transform(numeric_raw)
                # Extract scaled values
                age_sc, trestbps_sc, chol_sc, thalach_sc, oldpeak_sc = numeric_scaled[0]

                # Step 2 — One-hot encode categoricals (drop_first=True, matching training)
                # cp: 0,1,2,3 → cp_1, cp_2, cp_3 (drop cp_0)
                cp_1 = 1 if cp_h == 1 else 0
                cp_2 = 1 if cp_h == 2 else 0
                cp_3 = 1 if cp_h == 3 else 0

                # slope: 0,1,2 → slope_1, slope_2 (drop slope_0)
                slope_1 = 1 if slope_h == 1 else 0
                slope_2 = 1 if slope_h == 2 else 0

                # thal: 0,1,2,3 → thal_1, thal_2, thal_3 (drop thal_0)
                thal_1 = 1 if thal_h == 1 else 0
                thal_2 = 1 if thal_h == 2 else 0
                thal_3 = 1 if thal_h == 3 else 0

                # Step 3 — Select the exact 10 features the model was trained on
                # (determined by SelectKBest during training)
                selected_input = pd.DataFrame(
                    [[sex_h, thalach_sc, exang_h, oldpeak_sc, ca_h,
                      cp_2, slope_1, slope_2, thal_2, thal_3]],
                    columns=['sex', 'thalach', 'exang', 'oldpeak', 'ca',
                             'cp_2', 'slope_1', 'slope_2', 'thal_2', 'thal_3']
                )

                # Step 4 — Predict cluster directly (KMeans was trained on these 10 features)
                cluster = int(kmeans.predict(selected_input)[0])
            except Exception as exc:
                st.error(
                    "❌ **Processing error** — the input dimensions may not match the "
                    "training pipeline.  \n\n"
                    f"**Details:** `{exc}`"
                )
                st.stop()

        st.markdown("---")
        st.subheader("🔍 Segmentation Results")

        # Cluster descriptions (edit these to match your analysis)
        CLUSTER_INFO = {
            0: {
                "label": "Low Risk Profile",
                "emoji": "🟢",
                "css": "cluster-low",
                "description": (
                    "Patient presents **stable cardiovascular vitals** with low cardiac "
                    "stress indicators.  Continue routine check-ups."
                ),
                "recommendation": "Maintain a heart-healthy diet and regular exercise routine.",
            },
            1: {
                "label": "Moderate Risk / At-Risk",
                "emoji": "🟡",
                "css": "cluster-moderate",
                "description": (
                    "Patient exhibits **elevated cholesterol or blood pressure**. "
                    "Closer monitoring and lifestyle adjustments are recommended."
                ),
                "recommendation": (
                    "Schedule a follow-up lipid panel and BP monitoring within 3 months."
                ),
            },
            2: {
                "label": "High Risk Profile",
                "emoji": "🔴",
                "css": "cluster-high",
                "description": (
                    "Patient shows **strong correlation with ST depression and exercise-induced "
                    "angina**. Immediate cardiology referral is advised."
                ),
                "recommendation": (
                    "Refer to a cardiologist for advanced diagnostic imaging (e.g., stress echo, angiography)."
                ),
            },
        }

        info = CLUSTER_INFO.get(
            cluster,
            {
                "label": f"Cluster {cluster}",
                "emoji": "⚪",
                "css": "cluster-moderate",
                "description": "No predefined interpretation for this cluster.",
                "recommendation": "Please consult your data scientist to label this cluster.",
            },
        )

        c1, c2 = st.columns([1, 2])

        with c1:
            st.markdown(
                f'<div class="cluster-card {info["css"]}">'
                f'<h2>{info["emoji"]} Cluster {cluster}</h2>'
                f'<h4>{info["label"]}</h4>'
                "</div>",
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(f"**Interpretation:** {info['description']}")
            st.info(f"💡 **Recommendation:** {info['recommendation']}")

        # Show input summary
        with st.expander("📋 View submitted patient data"):
            raw_values = [
                age_h, sex_h, cp_h, trestbps_h, chol_h, fbs_h,
                restecg_h, thalach_h, exang_h, oldpeak_h, slope_h, ca_h, thal_h,
            ]
            feature_names = [
                "Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
                "Fasting BS > 120", "Resting ECG", "Max Heart Rate",
                "Exercise Angina", "ST Depression", "Slope", "Major Vessels",
                "Thalassemia",
            ]
            for name, val in zip(feature_names, raw_values):
                st.write(f"- **{name}:** {val}")