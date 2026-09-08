import streamlit as st

st.set_page_config(page_title="Credit Risk Assessment — Prototype", page_icon="🏦", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
    .stApp { background-color: #F6F4EE; }
    h1, h2, h3 { font-family: 'Georgia', serif; }
    .band-low { background:#E4EFE7; color:#2F6F4E; padding:6px 14px; font-weight:600; display:inline-block; border-radius:2px; }
    .band-med { background:#F3EAD3; color:#B8862C; padding:6px 14px; font-weight:600; display:inline-block; border-radius:2px; }
    .band-high { background:#F3E0DB; color:#A23B2E; padding:6px 14px; font-weight:600; display:inline-block; border-radius:2px; }
    .footnote { font-size:12.5px; color:#5B6272; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

st.caption("STUDENT PROJECT — NOT AFFILIATED WITH ICICI BANK")
st.title("AI-based credit risk assessment")
st.write("A working prototype of a machine-learning-assisted loan approval tool. "
         "Enter an applicant's profile and run the assessment to see a risk score, "
         "a recommendation, and the factors driving it.")

st.divider()

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
        employment = st.selectbox(
            "Employment type",
            ["Salaried — government/PSU", "Salaried — private sector",
             "Self-employed professional", "Business owner"],
            index=1
        )
        history = st.number_input("Credit history length (years)", min_value=0, value=4, step=1)
    with c4:
        loans = st.number_input("Existing active loans", min_value=0, value=1, step=1)
        ontime = st.slider("On-time repayment rate (%)", 0, 100, 94)

    run = st.button("Run risk assessment", type="primary", use_container_width=True)

# ---------- LOGIC ----------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def assess(income, loan_amt, emi, tenure, score, employment, history, loans, ontime):
    monthly_income = income / 12
    est_monthly_emi = (loan_amt / tenure) * 1.06 if tenure > 0 else loan_amt
    obligation_ratio = (emi + est_monthly_emi) / monthly_income if monthly_income > 0 else 1

    emp_weight = {
        "Salaried — government/PSU": 0.95,
        "Salaried — private sector": 0.80,
        "Self-employed professional": 0.60,
        "Business owner": 0.55,
    }[employment]

    c_bureau = clamp((900 - score) / 6, 0, 100) * 0.30
    c_obligation = clamp(obligation_ratio * 120, 0, 100) * 0.25
    c_income = clamp(100 - (monthly_income / 900), 0, 100) * 0.15
    c_employment = clamp((1 - emp_weight) * 100, 0, 100) * 0.12
    c_history = clamp(100 - history * 12, 0, 100) * 0.10
    c_repayment = clamp(100 - ontime, 0, 100) * 0.08

    raw = c_bureau + c_obligation + c_income + c_employment + c_history + c_repayment
    risk_score = int(clamp(round(raw), 2, 98))

    factors = [
        ("Bureau score", c_bureau, 30),
        ("Debt-to-income", c_obligation, 25),
        ("Income adequacy", c_income, 15),
        ("Employment type", c_employment, 12),
        ("Credit history", c_history, 10),
        ("Repayment track record", c_repayment, 8),
    ]
    factors.sort(key=lambda f: f[1], reverse=True)

    if risk_score < 35:
        band, css, decision, note = (
            "LOW RISK", "band-low", "Recommend: Approve",
            "Profile fits standard approval criteria. Route to auto-approval queue with routine documentation checks."
        )
    elif risk_score < 65:
        band, css, decision, note = (
            "MEDIUM RISK", "band-med", "Recommend: Refer for review",
            "Mixed signals — a credit officer should verify income and obligations before a final decision."
        )
    else:
        band, css, decision, note = (
            "HIGH RISK", "band-high", "Recommend: Decline / manual underwriting",
            "Elevated risk of default under current terms. Consider a smaller loan amount, a co-applicant, or additional collateral."
        )

    return risk_score, band, css, decision, note, factors

# ---------- OUTPUT ----------
with col_output:
    st.subheader("02 — Model output")

    if run:
        risk_score, band, css, decision, note, factors = assess(
            income, loan_amt, emi, tenure, score, employment, history, loans, ontime
        )

        top1, top2 = st.columns([1, 2])
        with top1:
            st.metric("Risk score (0–100)", risk_score)
        with top2:
            st.markdown(f'<span class="{css}">{band}</span>', unsafe_allow_html=True)
            st.markdown(f"**{decision}**")
            st.caption(note)

        st.markdown("**Factor contribution to risk score**")
        for name, val, maxv in factors:
            pct = int(round((val / maxv) * 100))
            st.write(name)
            st.progress(min(pct, 100) / 100)

    else:
        st.info("Fill in the applicant profile on the left and click **Run risk assessment** to see the score, decision, and factor breakdown.")

st.divider()
st.markdown(
    '<p class="footnote"><b>How to read this prototype:</b> the score is produced by a transparent '
    'weighted scorecard (income adequacy, obligation-to-income ratio, bureau score, employment stability, '
    'credit history depth, repayment track record) — not a live bank model. It demonstrates the decision '
    'logic a real ML classifier (e.g. gradient-boosted trees or logistic regression trained on historical '
    'loan performance) would learn from data. In production, automated declines should still route to '
    'human review, per fair-lending and explainability practice.</p>',
    unsafe_allow_html=True
)
