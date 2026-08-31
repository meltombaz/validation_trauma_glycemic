# ============================================================
# Streamlit app — Diabetes / Prediabetes Screening (reduced features)
# Single app: choose sex at the top, form + model switch accordingly.
#
# Run with:  streamlit run app_diabetes_screening.py
# Expects:   final_reduced_model_Female.pkl and final_reduced_model_Male.pkl
#            (from build_final_reduced_models.py) in the same directory,
#            OR set MODEL_DIR below / via the MODEL_DIR environment variable.
#
# UNITS/LABELS: taken from what you specified. One exception — APTT's
# unit was not given (only used by the Female model), so it defaults to
# seconds (the standard clinical unit) below. Verify this against your
# lab's reporting convention before relying on the app; the field is
# marked "(unit unverified)" in the UI as a reminder.
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = os.environ.get("MODEL_DIR", ".")
MODEL_PATHS = {
    "Female": os.path.join(MODEL_DIR, "final_reduced_model_Female.pkl"),
    "Male": os.path.join(MODEL_DIR, "final_reduced_model_Male.pkl"),
}

# Per-feature display info: label, unit, and categorical code -> label
# mapping. Features not listed here fall back to their raw column name
# and, for categorical fields, their raw observed codes.
FEATURE_INFO = {
    "Age": {"label": "Age", "unit": "years"},
    "LEU": {"label": "Leukocytes (Leukozyten)", "unit": "1/\u00b5L"},
    "MCHC": {"label": "MCHC", "unit": "g/dL"},
    "MeanPlateletVolume": {"label": "Mean Platelet Volume", "unit": "fL"},
    "QUICK": {"label": "QUICK", "unit": "%"},
    "WaistCircumference": {"label": "Waist Circumference", "unit": "cm"},
    "APTT": {"label": "APTT", "unit": "sec (unit unverified)"},
    "BMI": {"label": "BMI", "unit": "kg/m\u00b2", "computed_from_weight_height": True},
    "Previous High Blood Sugar Levels": {
        "label": "Previous High Blood Sugar Levels",
        "categorical_labels": {2: "Yes", 0: "No"},
    },
    "High Blood Pressure Medicine": {
        "label": "High Blood Pressure Medicine",
        "categorical_labels": {2: "Yes", 0: "No"},
    },
}

st.set_page_config(page_title="Diabetes Screening", layout="centered")


@st.cache_resource
def load_artifact(path):
    return joblib.load(path)


def field_label(feat):
    info = FEATURE_INFO.get(feat, {})
    label = info.get("label", feat)
    unit = info.get("unit")
    return f"{label} ({unit})" if unit else label


def build_input_row(artifact):
    feature_names = artifact["feature_names"]
    feature_stats = artifact["feature_stats"]
    values = {}

    st.subheader("Patient data")
    for feat in feature_names:
        info = FEATURE_INFO.get(feat, {})
        stats = feature_stats[feat]

        if info.get("computed_from_weight_height"):
            st.markdown(f"**{field_label(feat)}** — calculated from weight and height")
            c1, c2 = st.columns(2)
            weight_kg = c1.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=75.0, step=0.5)
            height_cm = c2.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0, step=0.5)
            bmi = weight_kg / ((height_cm / 100) ** 2)
            st.caption(f"Calculated BMI: {bmi:.1f} kg/m\u00b2 "
                       f"(training data range: {stats['min']:.1f} - {stats['max']:.1f})")
            values[feat] = bmi
            continue

        if stats["type"] == "numeric":
            values[feat] = st.number_input(
                field_label(feat),
                min_value=float(stats["min"]),
                max_value=float(stats["max"]) * 1.5 if stats["max"] > 0 else float(stats["max"]),
                value=float(stats["median"]),
                help=f"Training data range: {stats['min']:.2f} - {stats['max']:.2f}",
            )
        else:  # categorical
            raw_options = stats["values"]
            labels = info.get("categorical_labels", {})
            display = [f"{labels.get(v, v)}" for v in raw_options]
            choice_display = st.selectbox(field_label(feat), display)
            values[feat] = raw_options[display.index(choice_display)]

    return pd.DataFrame([values])[feature_names]


def predict(artifact, X_row):
    X = X_row.copy()
    num_in = artifact["num_features"]
    cat_in = artifact["cat_features"]

    if num_in and artifact["num_imputer"] is not None:
        X[num_in] = artifact["num_imputer"].transform(X[num_in])
        if artifact["scaler"] is not None:
            X[num_in] = artifact["scaler"].transform(X[num_in])
    if cat_in and artifact["cat_imputer"] is not None:
        X[cat_in] = artifact["cat_imputer"].transform(X[cat_in])

    X = X[artifact["feature_names"]]
    proba = float(artifact["model"].predict_proba(X)[:, 1][0])
    return proba


def main():
    st.title("Diabetes / Prediabetes Screening")
    st.caption(
        "Research tool for internal validation only — not a clinical diagnostic device. "
        "Predicts risk of Pre-diabetes/Diabetes (Group 1/2) vs. Normal (Group 0)."
    )

    sex = st.radio("Sex", list(MODEL_PATHS.keys()), horizontal=True)
    model_path = MODEL_PATHS[sex]

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}. Set the MODEL_DIR environment "
                 f"variable, or place both .pkl files next to this script.")
        return

    artifact = load_artifact(model_path)

    with st.expander("Model info"):
        st.write(f"Sex stratum: {artifact['sex_stratum']}")
        st.write(f"Features used ({len(artifact['feature_names'])}): {artifact['feature_names']}")
        st.write(f"Tuning CV AUC (5-fold, full stratum data): {artifact['cv_auc_mean']:.4f}")
        st.caption(
            "This is the tuning-time CV AUC on the deployment model, not the reported "
            "generalization estimate — use the isolated nested-CV fold AUCs from your "
            "manuscript for that."
        )

    X_row = build_input_row(artifact)

    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

    if st.button("Predict"):
        proba = predict(artifact, X_row)
        pred_label = "Pre-diabetes / Diabetes risk" if proba >= threshold else "Normal"

        st.subheader("Result")
        st.metric("Predicted probability (Pre-/DM)", f"{proba:.3f}")
        st.write(f"Classification at threshold {threshold:.2f}: **{pred_label}**")
        st.progress(min(max(proba, 0.0), 1.0))

        with st.expander("Input used for this prediction"):
            st.dataframe(X_row.T.rename(columns={0: "value"}))


if __name__ == "__main__":
    main()
