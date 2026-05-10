import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


def load_pickle_file(file_path: Path):
    """Load a pickle file with explicit error handling."""
    with file_path.open("rb") as f:
        return pickle.load(f)


def label_encode_value(value: str, choices: list[str]) -> int:
    """
    Encode a categorical value using LabelEncoder-like alphabetical behavior.
    Equivalent to fitting LabelEncoder on `choices` and transforming `value`.
    """
    sorted_classes = sorted(set(choices))
    mapping = {cls: idx for idx, cls in enumerate(sorted_classes)}
    if value not in mapping:
        raise ValueError(f"Value '{value}' not found in categorical mapping.")
    return mapping[value]


def build_feature_row(inputs: dict, expected_columns: list[str]) -> pd.DataFrame:
    """Create one-row dataframe ordered exactly as training columns."""
    df = pd.DataFrame([inputs])
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[expected_columns]


def get_risk_level(churn_probability: float) -> str:
    """Return textual risk level from churn probability."""
    if churn_probability >= 0.60:
        return "🔴 High Risk"
    if churn_probability >= 0.30:
        return "🟡 Medium Risk"
    return "🟢 Low Risk"


def get_recommendation(churn_probability: float) -> str:
    """Return business recommendation based on churn risk."""
    if churn_probability > 0.60:
        return "Offer loyalty discount or premium plan upgrade"
    if churn_probability >= 0.30:
        return "Send re-engagement email with special offer"
    return "Customer is stable. Maintain service quality"


def main() -> None:
    st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

    st.markdown(
        """
        <style>
        .result-box {
            border-radius: 10px;
            padding: 18px 20px;
            border: 2px solid;
            margin-bottom: 12px;
        }
        .result-green {
            background: #d4edda;
            border-color: #28a745;
            color: #155724;
        }
        .result-red {
            background: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }
        .subtitle {
            color: #6c757d;
            margin-top: -6px;
            margin-bottom: 10px;
        }
        .insight-card {
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 14px;
            background: #ffffff;
            min-height: 150px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📊 Customer Churn Predictor")
    st.markdown(
        '<p class="subtitle">Predict whether a customer will churn based on their service usage</p>',
        unsafe_allow_html=True,
    )
    st.info(
        "ℹ️ This model is trained on the IBM Telco Customer Churn Dataset (7,032 records) "
        "using Random Forest with 79.25% accuracy"
    )
    st.divider()

    model_path = Path("model.pkl")
    columns_path = Path("columns.pkl")

    model = None
    expected_columns = None
    model_error = None

    try:
        model = load_pickle_file(model_path)
        expected_columns = load_pickle_file(columns_path)
        if not isinstance(expected_columns, list):
            raise TypeError("`columns.pkl` must contain a list of column names.")
    except Exception as exc:
        model_error = f"Model loading error: {exc}"

    st.sidebar.header("Enter User Details")

    gender = st.sidebar.selectbox("gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox(
        "Senior Citizen (65+)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
    )
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Customer Tenure (months)", min_value=0, max_value=72, value=12)
    phone_service = st.sidebar.selectbox("Has Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Has Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.sidebar.selectbox("Has Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.sidebar.selectbox("Has Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("Has Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streams TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streams Movies", ["No", "Yes", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer automatic", "Credit card automatic"],
    )
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.01)
    total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0, step=0.01)

    predict_clicked = st.sidebar.button("🔍 Predict Churn", use_container_width=True)

    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_right:
        st.subheader("User Summary")
        summary_df = pd.DataFrame(
            {
                "Feature": [
                    "gender",
                    "SeniorCitizen",
                    "Partner",
                    "Dependents",
                    "PhoneService",
                    "MultipleLines",
                    "InternetService",
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                    "Contract",
                    "PaperlessBilling",
                    "PaymentMethod",
                ],
                "Value": [
                    gender,
                    "Yes" if senior_citizen == 1 else "No",
                    partner,
                    dependents,
                    phone_service,
                    multiple_lines,
                    internet_service,
                    online_security,
                    online_backup,
                    device_protection,
                    tech_support,
                    streaming_tv,
                    streaming_movies,
                    contract,
                    paperless_billing,
                    payment_method,
                ],
            }
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("tenure", f"{tenure} months")
        m2.metric("MonthlyCharges", f"${monthly_charges:,.2f}")
        m3.metric("TotalCharges", f"${total_charges:,.2f}")

    with col_left:
        st.subheader("Prediction Result")
        if model_error:
            st.error(model_error)
        elif predict_clicked:
            try:
                raw_input = {
                    "gender": label_encode_value(gender, ["Male", "Female"]),
                    "SeniorCitizen": int(senior_citizen),
                    "Partner": label_encode_value(partner, ["Yes", "No"]),
                    "Dependents": label_encode_value(dependents, ["Yes", "No"]),
                    "tenure": int(tenure),
                    "PhoneService": label_encode_value(phone_service, ["Yes", "No"]),
                    "MultipleLines": label_encode_value(multiple_lines, ["No", "Yes", "No phone service"]),
                    "InternetService": label_encode_value(internet_service, ["DSL", "Fiber optic", "No"]),
                    "OnlineSecurity": label_encode_value(online_security, ["No", "Yes", "No internet service"]),
                    "OnlineBackup": label_encode_value(online_backup, ["No", "Yes", "No internet service"]),
                    "DeviceProtection": label_encode_value(device_protection, ["No", "Yes", "No internet service"]),
                    "TechSupport": label_encode_value(tech_support, ["No", "Yes", "No internet service"]),
                    "StreamingTV": label_encode_value(streaming_tv, ["No", "Yes", "No internet service"]),
                    "StreamingMovies": label_encode_value(streaming_movies, ["No", "Yes", "No internet service"]),
                    "Contract": label_encode_value(contract, ["Month-to-month", "One year", "Two year"]),
                    "PaperlessBilling": label_encode_value(paperless_billing, ["Yes", "No"]),
                    "PaymentMethod": label_encode_value(
                        payment_method,
                        ["Electronic check", "Mailed check", "Bank transfer automatic", "Credit card automatic"],
                    ),
                    "MonthlyCharges": float(monthly_charges),
                    "TotalCharges": float(total_charges),
                }

                input_df = build_feature_row(raw_input, expected_columns)

                pred = int(model.predict(input_df)[0])
                proba = model.predict_proba(input_df)[0]

                churn_probability = float(proba[1])
                stay_probability = float(proba[0])
                confidence_value = churn_probability if pred == 1 else stay_probability

                if pred == 1:
                    st.markdown(
                        f"""
                        <div class="result-box result-red">
                            <h3 style="margin:0;">🔴 HIGH CHURN RISK</h3>
                            <p style="margin:8px 0 0 0; font-size:18px;">
                                Churn probability: <strong>{churn_probability * 100:.2f}%</strong>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-box result-green">
                            <h3 style="margin:0;">🟢 LIKELY TO STAY</h3>
                            <p style="margin:8px 0 0 0; font-size:18px;">
                                Stay probability: <strong>{stay_probability * 100:.2f}%</strong>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.write("Confidence Meter")
                st.progress(int(np.clip(confidence_value * 100, 0, 100)))

                st.divider()
                c1, c2, c3 = st.columns(3)
                risk_level = get_risk_level(churn_probability)
                recommendation = get_recommendation(churn_probability)

                with c1:
                    with st.container(border=True):
                        st.markdown("### 📊 Top Churn Factors")
                        st.markdown("- TotalCharges (most important)")
                        st.markdown("- MonthlyCharges")
                        st.markdown("- Customer Tenure")
                with c2:
                    with st.container(border=True):
                        st.markdown("### ⚠️ Risk Level")
                        st.markdown(f"**{risk_level}**")
                        st.markdown(f"Churn Probability: **{churn_probability * 100:.2f}%**")
                with c3:
                    with st.container(border=True):
                        st.markdown("### 💡 Recommendation")
                        st.markdown(recommendation)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
        else:
            st.info("Enter customer details in the sidebar and click `🔍 Predict Churn`.")

    st.markdown("---")
    st.caption("Customer Churn Predictor | Data Science Portfolio Project | 2025")


if __name__ == "__main__":
    main()
