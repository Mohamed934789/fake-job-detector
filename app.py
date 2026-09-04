"""
app.py — Fake Job Posting Detector
------------------------------------
Streamlit deployment for the Stacking Ensemble trained in train_model.py.

Run with:
    streamlit run app.py

Requires "fake_job_stacking_model.joblib" (produced by train_model.py) to be
present in the same folder.
"""

import os
import re

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from feature_engineering import JobPostingFeatureEngineer  # noqa: F401  (needed for unpickling)

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Fake Job Posting Detector · Samsung Innovation Campus",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "fake_job_stacking_model.joblib"

# ----------------------------------------------------------------------
# Custom CSS — dark glassmorphism theme
# ----------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 20% 0%, #0B1E6B 0%, #060B24 45%, #04061a 100%);
    color: #EAF0FF;
}

h1, h2, h3, .hero-title {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Samsung wordmark badge */
.samsung-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    background: rgba(20, 40, 160, 0.35);
    border: 1px solid rgba(41, 121, 255, 0.55);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    letter-spacing: 0.06em;
    font-size: 0.78rem;
    color: #7DB3FF;
    margin-bottom: 0.9rem;
}
.samsung-badge b {
    color: #FFFFFF;
    font-weight: 700;
}

/* Hero header */
.hero {
    padding: 2.4rem 2.2rem;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(20,40,160,0.55), rgba(0,169,255,0.14));
    border: 1px solid rgba(66, 133, 244, 0.25);
    margin-bottom: 1.6rem;
    box-shadow: 0 20px 60px rgba(20, 40, 160, 0.35);
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7DB3FF, #00A9FF 55%, #A8D8FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-sub {
    color: #B7C6EF;
    font-size: 1.02rem;
    max-width: 720px;
}

/* Glass cards */
.glass-card {
    background: rgba(66, 133, 244, 0.055);
    border: 1px solid rgba(125, 179, 255, 0.14);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}

.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    background: rgba(0, 169, 255, 0.16);
    border: 1px solid rgba(0, 169, 255, 0.4);
    font-size: 0.85rem;
    color: #B9E1FF;
    margin-right: 0.5rem;
}

.verdict-safe {
    background: linear-gradient(135deg, rgba(16,185,129,0.22), rgba(16,185,129,0.05));
    border: 1px solid rgba(16,185,129,0.45);
    color: #6EE7B7;
}
.verdict-danger {
    background: linear-gradient(135deg, rgba(244,63,94,0.25), rgba(244,63,94,0.05));
    border: 1px solid rgba(244,63,94,0.5);
    color: #FCA5A5;
}
.verdict-box {
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    font-size: 1.15rem;
    font-weight: 600;
    text-align: center;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1440 0%, #04061a 100%);
    border-right: 1px solid rgba(66, 133, 244, 0.18);
}

.stButton>button {
    background: linear-gradient(90deg, #1428A0, #00A9FF);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.4rem;
    font-weight: 600;
    font-size: 1rem;
    box-shadow: 0 10px 30px rgba(20, 40, 160, 0.45);
    transition: transform 0.15s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
}

hr { border-color: rgba(125, 179, 255, 0.14); }

.sidebar-footer-badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 0.8rem;
    border-radius: 12px;
    background: rgba(20, 40, 160, 0.3);
    border: 1px solid rgba(0, 169, 255, 0.3);
    font-size: 0.78rem;
    color: #9FC4FF;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


model = load_model(MODEL_PATH)

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="samsung-badge">🔷 <b>SAMSUNG</b> Innovation Campus</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### 🕵️ Fake Job Detector")
    st.caption("Stacking Ensemble · TF-IDF + Metadata")
    page = st.radio(
        "Navigate",
        ["🔍 Check a Job Posting", "📂 Batch Check (CSV)", "📊 Model Performance"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
        **Model:** Stacking Ensemble
        (Logistic Regression + Linear SVC + Random Forest + XGBoost,
        meta-learner: Logistic Regression)

        **Test F1-score:** 0.850
        **Test ROC-AUC:** 0.983
        """
    )
    st.markdown("---")
    st.markdown(
        """
        <div class="sidebar-footer-badge">
            🔷 Built for <b style="color:#EAF0FF;">Samsung Innovation Campus (SIC)</b> — AI Track
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Built with Streamlit · scikit-learn · imbalanced-learn · XGBoost")

# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
    <div class="samsung-badge">🔷 <b>SAMSUNG</b> Innovation Campus &nbsp;·&nbsp; AI Track</div>
    <div class="hero-title">Fake Job Posting Detector</div>
    <div class="hero-sub">
        Paste a job listing and the model — a Stacking Ensemble trained on
        17,880 real-world postings — estimates the probability that it's
        fraudulent, based on its text and metadata.
    </div>
    <div style="margin-top:1rem;">
        <span class="metric-pill">🎯 F1 0.850</span>
        <span class="metric-pill">📈 ROC-AUC 0.983</span>
        <span class="metric-pill">🧠 4 base models + meta-learner</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        f"⚠️ Model file **{MODEL_PATH}** was not found next to app.py.\n\n"
        "Run `python train_model.py` first (with `fake_job_postings.csv` in the same "
        "folder) to generate it, then restart this app."
    )
    st.stop()


def gauge_chart(probability):
    color = "#F43F5E" if probability >= 0.5 else "#10B981"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 40, "color": "#EAF0FF"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#5C79C2"},
                "bar": {"color": color},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(16,185,129,0.18)"},
                    {"range": [30, 60], "color": "rgba(250,204,21,0.18)"},
                    {"range": [60, 100], "color": "rgba(244,63,94,0.18)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#EAF0FF"},
    )
    return fig


def render_result(row, note=None):
    """Shared result renderer: runs the model on one row and shows the gauge + verdict."""
    proba = model.predict_proba(row)[0, 1]
    pred = int(proba >= 0.5)

    st.plotly_chart(gauge_chart(proba), use_container_width=True)

    if pred == 1:
        st.markdown(
            f'<div class="verdict-box verdict-danger">⚠️ Likely FRAUDULENT<br>'
            f'<span style="font-size:0.9rem; font-weight:400;">Estimated fraud probability: {proba:.1%}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="verdict-box verdict-safe">✅ Likely LEGITIMATE<br>'
            f'<span style="font-size:0.9rem; font-weight:400;">Estimated fraud probability: {proba:.1%}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if note:
        st.caption(note)
    st.caption(
        "This score reflects patterns learned from historical postings "
        "(missing company info, short descriptions, salary presence, etc.) "
        "and is a decision aid, not a certainty."
    )


def quick_parse(raw_text):
    """
    Turn one pasted blob of job-posting text into the fields the model
    expects. The free text always goes in fully (nothing is lost for the
    TF-IDF part). The 3 metadata flags (salary / company profile / location)
    are auto-detected with simple keyword rules since they aren't explicitly
    separated in a single pasted block.
    """
    raw_text = (raw_text or "").strip()
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    title_guess = lines[0][:120] if lines else ""

    salary_hit = bool(
        re.search(r"(\$|USD|EGP|LE|salary|راتب|مرتب)\s*[:\-]?\s*\d", raw_text, re.IGNORECASE)
    ) or bool(re.search(r"\d[\d,]{2,}\s*-\s*\d[\d,]{2,}", raw_text))

    company_hit = bool(
        re.search(
            r"(about us|about the company|company profile|who we are|our company|نبذة عن الشركة)",
            raw_text,
            re.IGNORECASE,
        )
    )

    location_hit = bool(
        re.search(r"(location|based in|remote|on-?site|hybrid|الموقع|المكان)\s*[:\-]?", raw_text, re.IGNORECASE)
    )

    logo_hit = bool(re.search(r"(logo|linkedin\.com/company|www\.)", raw_text, re.IGNORECASE))
    questions_hit = bool(re.search(r"\?", raw_text))

    return {
        "title": title_guess,
        "description": raw_text,
        "requirements": "",
        "benefits": "",
        "company_profile": "detected" if company_hit else None,
        "salary_range": "detected" if salary_hit else None,
        "location": "detected" if location_hit else None,
        "department": None,
        "employment_type": "Unknown",
        "required_experience": "Unknown",
        "required_education": "Unknown",
        "industry": "Unknown",
        "function": "Unknown",
        "telecommuting": 0,
        "has_company_logo": int(logo_hit),
        "has_questions": int(questions_hit),
    }, {"salary": salary_hit, "company": company_hit, "location": location_hit}


# ----------------------------------------------------------------------
# PAGE 1 — Single prediction
# ----------------------------------------------------------------------
if page == "🔍 Check a Job Posting":
    tab_quick, tab_form = st.tabs(["⚡ Paste Full Text (one box)", "📝 Detailed Form"])

    # ---------------- Quick paste tab ----------------
    with tab_quick:
        col_form, col_result = st.columns([1.15, 1], gap="large")

        with col_form:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Paste the Whole Job Posting")
            st.caption(
                "Paste everything as one block — title, description, requirements, "
                "benefits, salary, company info. The app auto-detects salary / company "
                "profile / location mentions from the text."
            )
            raw_text = st.text_area(
                "Job posting text",
                height=380,
                placeholder=(
                    "Paste the full job ad here, e.g.:\n\n"
                    "Remote Data Entry Clerk\n\n"
                    "We are hiring urgently! Earn $3000-$5000/month working from home...\n"
                    "Requirements: no experience needed...\n"
                    "Benefits: flexible hours..."
                ),
                label_visibility="collapsed",
            )
            st.markdown("</div>", unsafe_allow_html=True)
            quick_analyze = st.button(
                "🔎 Analyze This Text", use_container_width=True, key="quick_btn"
            )

        with col_result:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Result")

            if quick_analyze:
                if not raw_text.strip():
                    st.warning("Paste some job posting text first.")
                else:
                    fields, detected = quick_parse(raw_text)
                    row = pd.DataFrame([fields])

                    detected_bits = [k for k, v in detected.items() if v]
                    note = (
                        "Auto-detected: " + ", ".join(detected_bits) + "."
                        if detected_bits
                        else "No salary / company-profile / location mentions were auto-detected in the text."
                    )
                    render_result(row, note=note)
            else:
                st.info("Paste the job posting on the left and click **Analyze This Text**.")

            st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Detailed form tab ----------------
    with tab_form:
        col_form, col_result = st.columns([1.15, 1], gap="large")

        with col_form:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Job Posting Details")

            title = st.text_input("Job Title", placeholder="e.g. Remote Data Entry Clerk")
            company_profile = st.text_area(
                "Company Profile", placeholder="About the company...", height=90
            )
            description = st.text_area(
                "Job Description", placeholder="Full job description...", height=140
            )
            requirements = st.text_area(
                "Requirements", placeholder="Skills, experience needed...", height=110
            )
            benefits = st.text_area("Benefits", placeholder="Perks, benefits...", height=90)

            c1, c2 = st.columns(2)
            with c1:
                location = st.text_input("Location", placeholder="e.g. New York, US")
                employment_type = st.selectbox(
                    "Employment Type",
                    ["Unknown", "Full-time", "Part-time", "Contract", "Temporary", "Other"],
                )
                required_experience = st.selectbox(
                    "Required Experience",
                    [
                        "Unknown",
                        "Internship",
                        "Entry level",
                        "Associate",
                        "Mid-Senior level",
                        "Director",
                        "Executive",
                        "Not Applicable",
                    ],
                )
                required_education = st.selectbox(
                    "Required Education",
                    [
                        "Unknown",
                        "High School or equivalent",
                        "Bachelor's Degree",
                        "Master's Degree",
                        "Doctorate",
                        "Some College Coursework Completed",
                        "Certification",
                        "Unspecified",
                    ],
                )
            with c2:
                salary_range = st.text_input("Salary Range (optional)", placeholder="e.g. 40000-60000")
                industry = st.text_input("Industry", placeholder="e.g. Oil & Energy")
                function = st.text_input("Function", placeholder="e.g. Administrative")
                department = st.text_input("Department", placeholder="e.g. Sales")

            c3, c4, c5 = st.columns(3)
            with c3:
                telecommuting = st.checkbox("Telecommuting")
            with c4:
                has_company_logo = st.checkbox("Has Company Logo", value=True)
            with c5:
                has_questions = st.checkbox("Has Screening Questions")

            st.markdown("</div>", unsafe_allow_html=True)

            analyze = st.button("🔎 Analyze Posting", use_container_width=True, key="form_btn")

        with col_result:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Result")

            if analyze:
                if not title and not description:
                    st.warning("Please enter at least a job title or description.")
                else:
                    row = pd.DataFrame(
                        [
                            {
                                "title": title,
                                "company_profile": company_profile,
                                "description": description,
                                "requirements": requirements,
                                "benefits": benefits,
                                "location": location,
                                "department": department,
                                "salary_range": salary_range if salary_range.strip() else None,
                                "employment_type": employment_type,
                                "required_experience": required_experience,
                                "required_education": required_education,
                                "industry": industry,
                                "function": function,
                                "telecommuting": int(telecommuting),
                                "has_company_logo": int(has_company_logo),
                                "has_questions": int(has_questions),
                            }
                        ]
                    )
                    render_result(row)
            else:
                st.info("Fill in the form and click **Analyze Posting** to get a fraud-risk score.")

            st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# PAGE 2 — Batch prediction
# ----------------------------------------------------------------------
elif page == "📂 Batch Check (CSV)":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Batch Check from CSV")
    st.write(
        "Upload a CSV with the same columns as the original dataset "
        "(`title`, `description`, `requirements`, `benefits`, `company_profile`, "
        "`location`, `salary_range`, `employment_type`, `required_experience`, "
        "`required_education`, `industry`, `function`, `department`, "
        "`telecommuting`, `has_company_logo`, `has_questions`). Missing columns are fine."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        with st.spinner("Scoring postings..."):
            probs = model.predict_proba(batch_df)[:, 1]
            batch_df["fraud_probability"] = probs
            batch_df["prediction"] = (probs >= 0.5).astype(int)
            batch_df["prediction_label"] = batch_df["prediction"].map(
                {1: "Fraudulent", 0: "Legitimate"}
            )

        st.success(f"Scored {len(batch_df)} postings.")

        c1, c2 = st.columns(2)
        c1.metric("Flagged as Fraudulent", int(batch_df["prediction"].sum()))
        c2.metric("Flagged as Legitimate", int((batch_df["prediction"] == 0).sum()))

        st.dataframe(
            batch_df.sort_values("fraud_probability", ascending=False),
            use_container_width=True,
            height=420,
        )

        csv_out = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results CSV", csv_out, "scored_job_postings.csv", "text/csv"
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# PAGE 3 — Model performance / about
# ----------------------------------------------------------------------
else:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Why the Stacking Ensemble?")
    st.write(
        "Several model families were benchmarked on the same train/test split "
        "using TF-IDF text features, categorical/metadata features, and class-imbalance "
        "handling (SMOTE / Random Over-Sampling). The Stacking ensemble — combining "
        "Logistic Regression, Linear SVC, Random Forest and XGBoost under a Logistic "
        "Regression meta-learner — gave the best overall balance of precision and recall."
    )

    perf = pd.DataFrame(
        [
            {"Model": "Stacking Ensemble", "Accuracy": 0.987, "Precision": 0.950, "Recall": 0.769, "F1": 0.850, "ROC-AUC": 0.983},
            {"Model": "Logistic Regression + ROS", "Accuracy": 0.984, "Precision": 0.866, "Recall": 0.786, "F1": 0.824, "ROC-AUC": 0.985},
            {"Model": "Hard Voting", "Accuracy": 0.984, "Precision": 0.992, "Recall": 0.676, "F1": 0.804, "ROC-AUC": None},
        ]
    )
    st.dataframe(perf, use_container_width=True, hide_index=True)

    fig = go.Figure()
    for metric, color in [("Precision", "#7C3AED"), ("Recall", "#38BDF8"), ("F1", "#F472B6")]:
        fig.add_trace(go.Bar(name=metric, x=perf["Model"], y=perf[metric], marker_color=color))
    fig.update_layout(
        barmode="group",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#EAF0FF"},
        legend=dict(orientation="h", y=1.15),
        margin=dict(t=20, l=10, r=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Pipeline Summary")
    st.markdown(
        """
        1. **Text cleaning** — lowercase, strip HTML/URLs/emails/punctuation.
        2. **Feature engineering** — combined text (`title + description + requirements + benefits`),
           text-length features, and information-availability flags
           (`has_salary`, `has_location`, `has_company_profile`, etc.).
        3. **Vectorization** — TF-IDF (unigrams + bigrams, 20,000 features) for text,
           StandardScaler for numeric features.
        4. **Class imbalance handling** — SMOTE / Random Over-Sampling on the training folds only.
        5. **Stacking** — 4 base learners feed a Logistic Regression meta-learner (5-fold CV).
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)