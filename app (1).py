import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Credit Risk Assessment — ML Prototype", page_icon="🏦", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
    .stApp { background-color: #F6F4EE; color: #1D2230; }
    h1, h2, h3, h4, p, span, label, div { color: #1D2230; }
    h1, h2, h3 { font-family: 'Georgia', serif; }
    .band-low { background:#E4EFE7; color:#2F6F4E; padding:6px 14px; font-weight:600; display:inline-block; border-radius:2px; }
    .band-med { background:#F3EAD3; color:#B8862C; padding:6px 14px; font-weight:600; display:inline-block; border-radius:2px; }
    .band-high { background:#F3E0DB; color:#A23B2E; padding:6px 14px; font-weight:600; display:inline-block; border-radius:2px; }
    .footnote { font-size:12.5px; color:#5B6272; line-height:1.6; }
    .push-up { color:#A23B2E; font-weight:600; }
    .push-down { color:#2F6F4E; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.caption("STUDENT PROJECT — NOT AFFILIATED WITH ICICI BANK — TRAINED ON SYNTHETIC DATA")
st.title("AI-based credit risk assessment")
st.write("This version uses a **trained machine learning model** (Random Forest), not a fixed rulebook. "
         "It predicts a probability of default learned from patterns in historical-style data, "
         "and explains which factors drove each individual prediction.")

st.divider()

FEATURES = ["income", "loan_amt", "emi", "tenure", "score", "employment_code",
            "history", "loans", "ontime"]

EMPLOYMENT_MAP = {
    "Salaried — government/PSU": 3,
    "Salaried — private sector": 2,
    "Self-employed professional": 1,
    "Business owner": 0,
}

# ---------- SYNTHETIC TRAINING DATA + MODEL (trained once, cached) ----------
@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    n = 6000

    income = rng.lognormal(mean=13.3, sigma=0.5, size=n)          # ~annual income, skewed
    loan_amt = rng.lognormal(mean=12.8, sigma=0.6, size=n)
    tenure = rng.integers(6, 84, size=n)
    emi = np.clip(rng.normal(income / 12 * 0.15, income / 12 * 0.08), 0, None)
    score = np.clip(rng.normal(680, 90, size=n), 300, 900)
    employment_code = rng.integers(0, 4, size=n)
    history = np.clip(rng.normal(5, 3, size=n), 0, 25)
    loans = rng.integers(0, 5, size=n)
    ontime = np.clip(rng.normal(88, 12, size=n), 0, 100)

    monthly_income = income / 12
    est_emi = (loan_amt / tenure) * 1.06
    dti = (emi + est_emi) / monthly_income

    log_income = np.log(income)
    z_income = (log_income - log_income.mean()) / log_income.std()
    z_score = (score - 650) / 90
    z_history = (history - 5) / 3
    z_ontime = (ontime - 85) / 12

    # ground-truth default probability generating process (noisy, nonlinear, with interactions).
    # Standardised (z-score) terms keep each factor's influence comparable, so no single
    # raw-magnitude feature (like income in rupees) swamps the others.
    logit = (
        -1.1
        + 1.8 * dti
        - 0.9 * z_score
        - 0.42 * employment_code
        - 0.35 * z_history
        - 0.3 * z_ontime
        + 0.22 * loans
        - 0.3 * z_income
        + 0.5 * dti * (employment_code == 0)      # business owners hurt more by high DTI
        + rng.normal(0, 0.5, size=n)               # noise
    )
    prob_default = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, np.clip(prob_default, 0.01, 0.99))

    X = pd.DataFrame({
        "income": income, "loan_amt": loan_amt, "emi": emi, "tenure": tenure,
        "score": score, "employment_code": employment_code, "history": history,
        "loans": loans, "ontime": ontime,
    })

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=20, random_state=42
    )
    model.fit(X[FEATURES], y)

    # population means, used later to compute each feature's individual contribution
    baseline = X[FEATURES].mean()
    return model, baseline

model, baseline = train_model()

DISPLAY_NAMES = {
    "income": "Annual income", "loan_amt": "Loan amount", "emi": "Existing EMI",
    "tenure": "Tenure", "score": "Bureau score", "employment_code": "Employment type",
    "history": "Credit history", "loans": "Existing loans", "ontime": "On-time repayment %",
}

col_input, col_output = st.columns([1, 1], gap="large")

# ---------- INPUT ----------
with col_input:
    st.subheader("01 — Applicant profile")

    c1, c2 = st.columns(2)
    with c1:
        income = st.number_input("Annual income (₹)", min_value=0, value=720000, step=10000)
        emi = st.number_input("Existing monthly obligations (₹)", min_value=0, value=8000, step=500)
    with c2:
        loan_amt = st.number_input("Loan amount requested (₹)", min_value=0, value=500000, step=10000)
        tenure = st.number_input("Requested tenure (months)", min_value=3, value=36, step=1)

    score = st.slider("Credit bureau score (CIBIL-style, 300–900)", 300, 900, 712)

    c3, c4 = st.columns(2)
    with c3:
        employment = st.selectbox("Employment type", list(EMPLOYMENT_MAP.keys()), index=1)
        history = st.number_input("Credit history length (years)", min_value=0, value=4, step=1)
    with c4:
        loans = st.number_input("Existing active loans", min_value=0, value=1, step=1)
        ontime = st.slider("On-time repayment rate (%)", 0, 100, 94)

    run = st.button("Run risk assessment", type="primary", use_container_width=True)

# ---------- OUTPUT ----------
with col_output:
    st.subheader("02 — Model output")

    if run:
        x_input = pd.DataFrame([{
            "income": income, "loan_amt": loan_amt, "emi": emi, "tenure": tenure,
            "score": score, "employment_code": EMPLOYMENT_MAP[employment],
            "history": history, "loans": loans, "ontime": ontime,
        }])[FEATURES]

        prob_default = model.predict_proba(x_input)[0][1]
        risk_score = int(round(prob_default * 100))

        if risk_score < 30:
            band, css, decision = "LOW RISK", "band-low", "Recommend: Approve"
        elif risk_score < 60:
            band, css, decision = "MEDIUM RISK", "band-med", "Recommend: Refer for review"
        else:
            band, css, decision = "HIGH RISK", "band-high", "Recommend: Decline / manual underwriting"

        top1, top2 = st.columns([1, 2])
        with top1:
            st.metric("Predicted default probability", f"{risk_score}%")
        with top2:
            st.markdown(f'<span class="{css}">{band}</span>', unsafe_allow_html=True)
            st.markdown(f"**{decision}**")
            st.caption("Probability learned by the model from training data patterns — not a fixed rule threshold.")

        # ---------- LOCAL INTERPRETABILITY ----------
        # For each feature, swap in the population-average value and see how much
        # the predicted probability shifts. This isolates that feature's contribution
        # to THIS applicant's specific prediction.
        st.markdown("**Why the model predicted this — factor contributions**")
        st.caption("Each bar shows how much that factor pushed this applicant's risk up or down, versus a typical applicant.")

        contributions = []
        for feat in FEATURES:
            x_ablated = x_input.copy()
            x_ablated[feat] = baseline[feat]
            prob_ablated = model.predict_proba(x_ablated[FEATURES])[0][1]
            delta = (prob_default - prob_ablated) * 100  # positive = this factor raised risk
            contributions.append((DISPLAY_NAMES[feat], delta))

        contributions.sort(key=lambda t: abs(t[1]), reverse=True)
        max_abs = max(abs(d) for _, d in contributions) or 1

        for name, delta in contributions:
            direction = "push-up" if delta > 0 else "push-down"
            arrow = "▲ raises risk" if delta > 0 else "▼ lowers risk"
            width_pct = min(abs(delta) / max_abs * 100, 100)
            bar_color = "#A23B2E" if delta > 0 else "#2F6F4E"
            st.markdown(
                f"""<div style="margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:3px;">
                        <span>{name}</span>
                        <span class="{direction}">{arrow} ({delta:+.1f} pts)</span>
                    </div>
                    <div style="height:7px; background:#EDEAE0;">
                        <div style="height:100%; width:{width_pct}%; background:{bar_color};"></div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

    else:
        st.info("Fill in the applicant profile on the left and click **Run risk assessment** "
                "to see the predicted probability, decision, and factor explanation.")

st.divider()

with st.expander("Global model behaviour — what the model weighs most, across all applicants"):
    importances = pd.Series(model.feature_importances_, index=FEATURES).rename(DISPLAY_NAMES).sort_values(ascending=True)
    st.bar_chart(importances)
    st.caption("Feature importance from the trained Random Forest — how much each factor "
               "reduces prediction error on average, across the whole training set.")

st.markdown(
    '<p class="footnote"><b>About this prototype:</b> the model is a Random Forest classifier trained on '
    '6,000 synthetically generated applicant records (not real ICICI Bank data), so it demonstrates the '
    'modeling approach — learning nonlinear, interacting patterns from historical outcomes and explaining '
    'individual predictions — rather than production-accurate risk numbers. A real deployment would train '
    'on the bank\'s actual historical loan performance data, be validated for fairness across demographic '
    'groups, and route every automated decline to human review.</p>',
    unsafe_allow_html=True
)
