"""Streamlit web app — Credit Scoring System (browser UI).

Run with:
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import get_config
from src.inference.service import CreditScoringService

cfg = get_config()
ARTIFACT_PATH = Path(cfg["paths"]["models_dir"]) / "credit_scoring_logreg_v1.joblib"


@st.cache_resource
def get_service() -> CreditScoringService | None:
    """Load the scoring service once (cached across reruns)."""
    if not ARTIFACT_PATH.exists():
        return None
    return CreditScoringService(ARTIFACT_PATH)


def _applicant_form() -> dict | None:
    """Render the single-applicant input form; return a dict when submitted."""
    st.subheader("👤 Enter applicant details")
    with st.form("applicant_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age", 18, 100, 38)
            dependents = st.number_input("Dependents", 0, 20, 1)
            education = st.selectbox("Education", ["High School", "Bachelor", "Master", "Doctorate"], index=1)
            employment_status = st.selectbox("Employment status", ["Employed", "Self-Employed", "Unemployed"])
        with c2:
            employment_years = st.number_input("Employment years", 0.0, 50.0, 5.0)
            monthly_income = st.number_input("Monthly income", 0, 10_000_000, 50_000, step=1_000)
            monthly_expenses = st.number_input("Monthly expenses", 0, 10_000_000, 28_000, step=1_000)
            savings_balance = st.number_input("Savings balance", 0, 100_000_000, 1_000_000, step=10_000)
        with c3:
            credit_score = st.slider("Credit score (bureau)", 300, 850, 710)
            credit_utilization_ratio = st.slider("Credit utilization ratio", 0.0, 1.5, 0.35, 0.05)
            num_late_payments_12m = st.number_input("Late payments (last 12m)", 0, 12, 0)
            num_previous_defaults = st.number_input("Previous defaults", 0, 10, 0)
        with c4:
            loan_amount = st.number_input("Loan amount", 0, 100_000_000, 300_000, step=10_000)
            loan_term_months = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60], index=2)
            interest_rate = st.slider("Interest rate (%)", 0.0, 40.0, 9.0, 0.5)
            loan_purpose = st.selectbox(
                "Loan purpose",
                ["Debt Consolidation", "Home", "Auto", "Education", "Personal", "Business", "Medical"],
            )
        m1, m2 = st.columns(2)
        with m1:
            total_assets = st.number_input("Total assets", 0, 1_000_000_000, 3_000_000, step=100_000)
            monthly_debt_payment = st.number_input("Monthly debt payment", 0, 1_000_000, 8_000, step=500)
            num_open_accounts = st.number_input("Open credit accounts", 0, 50, 4)
        with m2:
            num_credit_inquiries_6m = st.number_input("Credit inquiries (last 6m)", 0, 30, 1)
            months_since_last_delinquency = st.number_input("Months since last delinquency", 0, 120, 0)
        submitted = st.form_submit_button("🔮 Predict creditworthiness", use_container_width=True, type="primary")

    if not submitted:
        return None
    return {
        "age": int(age), "dependents": int(dependents), "education": education,
        "employment_status": employment_status, "employment_years": float(employment_years),
        "monthly_income": float(monthly_income), "monthly_expenses": float(monthly_expenses),
        "savings_balance": float(savings_balance), "total_assets": float(total_assets),
        "monthly_debt_payment": float(monthly_debt_payment), "num_open_accounts": int(num_open_accounts),
        "num_credit_inquiries_6m": int(num_credit_inquiries_6m), "num_late_payments_12m": int(num_late_payments_12m),
        "num_previous_defaults": int(num_previous_defaults),
        "months_since_last_delinquency": float(months_since_last_delinquency),
        "credit_utilization_ratio": float(credit_utilization_ratio), "credit_score": int(credit_score),
        "interest_rate": float(interest_rate), "loan_amount": float(loan_amount),
        "loan_term_months": int(loan_term_months), "loan_purpose": loan_purpose,
    }


def _show_result(record: dict) -> None:
    """Render the prediction result + reason-code chart."""
    decision = record["decision"]
    color = "#16a34a" if decision == "Approve" else "#dc2626"
    emoji = "✅" if decision == "Approve" else "⛔"

    st.markdown(f"## {emoji} Decision: **{decision}**")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("P(creditworthy)", f"{record['p_creditworthy']:.1%}")
    col_b.metric("P(default)", f"{record['p_default']:.1%}")
    col_c.metric("Risk tier", record["risk_tier"])

    st.subheader("Top reason codes")
    reasons = pd.DataFrame(record["reasons"])
    if not reasons.empty:
        fig = go.Figure(go.Bar(
            x=reasons["contribution"], y=reasons["feature"], orientation="h",
            marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in reasons["contribution"]],
            text=reasons["direction"], textposition="auto",
        ))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Effect on approval (log-odds)")
        st.plotly_chart(fig, use_container_width=True)


def _batch_tab(service: CreditScoringService) -> None:
    st.subheader("📂 Batch scoring from CSV")
    st.caption("Upload a CSV with the 21 applicant features (same columns as the training data).")
    sample_path = Path(cfg["paths"]["processed_data"]).parent / "test.csv"
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    col_load, _ = st.columns([1, 3])
    with col_load:
        if st.button("Use sample (first 10 from test set)") and sample_path.exists():
            uploaded = None
            sample = pd.read_csv(sample_path).head(10)
            from src.preprocessing.pipeline import prepare_features
            X, _ = prepare_features(sample, target=cfg["dataset"]["target"],
                                    protected=cfg.get("protected_attributes", []))
            results = service.predict(X)
            st.dataframe(results, use_container_width=True)

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            results = service.predict(df)
            st.success(f"Scored {len(results)} applicants.")
            st.dataframe(results, use_container_width=True)
            st.download_button("⬇️ Download results (CSV)", results.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Scoring failed: {exc}")


def _info_tab(service: CreditScoringService) -> None:
    st.subheader("ℹ️ Model information")
    info = service.health_check()
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", info["model_type"].split("(")[0])
    c2.metric("Threshold", f"{info['threshold']:.2f}")
    c3.metric("Features", info["n_features"])
    test = info.get("test_metrics") or {}
    if test:
        st.markdown("#### Test-set performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", f"{test.get('roc_auc', 0):.3f}")
        m2.metric("KS", f"{test.get('ks', 0):.3f}")
        m3.metric("Brier", f"{test.get('brier', 0):.3f}")
        m4.metric("Cost/applicant", f"{test.get('cost_per_applicant', 0):.3f}")


def main() -> None:
    st.set_page_config(page_title="Credit Scoring System", page_icon="🏦", layout="wide")
    st.title("🏦 Credit Scoring System")
    st.caption("Production-ready creditworthiness prediction • cost-optimal, explainable decisions")

    service = get_service()
    if service is None:
        st.error("Trained model not found! Run this first:  `python -m src.inference.finalize`")
        st.stop()

    tab_single, tab_batch, tab_info = st.tabs(["👤 Single Applicant", "📂 Batch (CSV)", "ℹ️ Model Info"])

    with tab_single:
        applicant = _applicant_form()
        if applicant is not None:
            try:
                record = service.predict_single(applicant)
                _show_result(record)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")

    with tab_batch:
        _batch_tab(service)

    with tab_info:
        _info_tab(service)


if __name__ == "__main__":
    main()
