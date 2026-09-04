<div align="center">

# 🕵️ Fake Job Posting Detector

### AI-powered fraud detection for online job listings — built for **Samsung Innovation Campus (SIC)**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-0B6E4F?style=for-the-badge)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](#license)

**Paste any job posting → get an instant fraud-risk score, backed by a Stacking Ensemble trained on 17,880 real-world listings.**

[Live Demo](#-live-demo) · [How It Works](#-how-it-works) · [Model Results](#-model-results) · [Run Locally](#-run-it-locally) · [Deploy It](#-deploy-your-own-copy)

</div>

---

## 📌 Overview

Online job boards are a common target for scammers — fake "remote data entry" listings, too-good-to-be-true salaries, and postings with no real company behind them. This project builds a machine learning system that reads a job posting's text and metadata and predicts whether it's **legitimate** or **fraudulent**.

The repo contains two parts:

| Part | What it is |
|---|---|
| 📓 **`final_fake_job_project.ipynb`** | The full research notebook — data cleaning, EDA, and **5 experiments** comparing classical ML, resampling strategies, ensembling, and a transformer-based (DistilBERT) approach. |
| 🚀 **Deployment app** (`app.py`, `train_model.py`, `feature_engineering.py`) | A production-ready **Streamlit** web app serving the winning model, with a live fraud-probability gauge and batch CSV scoring. |

---

## 🖥️ Live Demo

> Add your Streamlit Community Cloud link here once deployed:
>
> **🔗 https://your-app-name.streamlit.app**

---

## 🧠 The Dataset

- **17,880 job postings** (based on the widely-used "Real / Fake Job Postings" dataset).
- Only **4.8%** (866 postings) are fraudulent — a heavily **imbalanced** classification problem.
- Fields include: title, description, requirements, benefits, company profile, salary range, location, employment type, required experience/education, industry, function, and boolean flags (telecommuting, has company logo, has screening questions).

### Key EDA findings
- 🚩 **Missing company profile is the strongest red flag** — present in 84% of real postings but only 32% of fake ones.
- 🚩 Fake postings mention a **salary range more often** (25.8% vs 15.5%) — enticing numbers are a classic scam tactic.
- 🚩 Fake postings tend to have **shorter, less detailed** descriptions and requirements.

---

## 🔬 Experiments — 5 Approaches Compared

| # | Experiment | Approach | Best F1-score |
|---|---|---|---|
| EX1 | Classical ML | All features (text + categorical + numeric) | 0.720 |
| EX2 | Simplified text | Single merged text field | 0.822 |
| EX3 | Imbalance handling | SMOTE / Random Over-Sampling per model | 0.826 |
| **EX4** | **Stacking Ensemble** | **4 base models → meta-learner** | **0.850 🏆** |
| EX5 | Deep Learning | DistilBERT embeddings + classical heads | 0.773 |

> 🏆 **EX4 (Stacking Ensemble) is the winner** and the model deployed in the app.

### Why Stacking won
Instead of a simple majority vote, a **Logistic Regression meta-learner** is trained on top of 4 base models' outputs — learning *when to trust which model* rather than weighting them all equally:

```
                ┌──────────────────────┐
   Job Posting  │   Feature Engineer    │
   (raw text +  │  clean → text_context │
   metadata)    │  + numeric flags      │
                └───────────┬───────────┘
                            │
         ┌──────────────────┼──────────────────┬──────────────────┐
         │                  │                  │                  │
 ┌───────▼──────┐  ┌────────▼───────┐  ┌───────▼──────┐  ┌────────▼───────┐
 │ Logistic Reg. │  │  Linear SVC    │  │ Random Forest│  │    XGBoost     │
 │   + ROS       │  │    + ROS       │  │   + SMOTE    │  │    + SMOTE     │
 └───────┬──────┘  └────────┬───────┘  └───────┬──────┘  └────────┬───────┘
         │                  │                  │                  │
         └──────────────────┴────────┬─────────┴──────────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │   Logistic Regression     │
                         │      (meta-learner)       │
                         └────────────┬────────────┘
                                      │
                           Fraud Probability (0–1)
```

---

## 📊 Model Results

**Final model — Stacking Ensemble (test set):**

| Metric | Score |
|---|---|
| Accuracy | **98.7%** |
| Precision | **95.0%** |
| Recall | **76.9%** |
| **F1-score** | **0.850** |
| ROC-AUC | **98.3%** |

<details>
<summary>📈 What do Precision, Recall, and F1 mean here?</summary>

- **Precision** — of everything flagged "fraudulent," how many actually are? (avoids falsely accusing real postings)
- **Recall** — of all real fraud cases, how many did the model catch?
- **F1-score** — the balance between the two; the metric that matters most for an imbalanced fraud problem.

</details>

---

## ⚙️ How It Works

### 🏋️ Training pipeline (run once, offline)
1. **Text cleaning** — lowercase, strip HTML/URLs/emails/punctuation.
2. **Feature engineering** — combined text (`title + description + requirements + benefits`), text-length features, and information-availability flags (`has_salary`, `has_location`, `has_company_profile`, etc.).
3. **Vectorization** — TF-IDF (unigrams + bigrams, 20,000 features) for text, StandardScaler for numeric features.
4. **Class imbalance handling** — SMOTE / Random Over-Sampling **on the training folds only**.
5. **Stacking** — 4 base learners feed a Logistic Regression meta-learner (5-fold CV).
6. Everything is saved as **one file**: `fake_job_stacking_model.joblib`.

### ⚡ Prediction pipeline (what happens live, in the app)
1. User pastes/enters a job posting.
2. The **same** feature-engineering class transforms it (no retraining — the TF-IDF vocabulary and scaler are already fitted).
3. **No SMOTE/ROS here** — that's a training-only balancing trick; a single new posting doesn't need resampling.
4. The 4 base models each output a probability → the meta-learner combines them into one final fraud probability.
5. Streamlit renders it as a gauge + a clear verdict.

---

## 📁 Repository Structure

```
.
├── final_fake_job_project.ipynb     # Full research notebook (EDA + 5 experiments)
├── feature_engineering.py           # Shared cleaning/feature logic (train + inference)
├── train_model.py                   # Rebuilds & saves the winning Stacking pipeline
├── app.py                           # Streamlit deployment app
├── requirements.txt                 # Python dependencies
├── fake_job_stacking_model.joblib   # Trained model (generated by train_model.py)
└── README.md
```

---

## 🚀 Run It Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (needs fake_job_postings.csv in this folder)
python train_model.py

# 4. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### App features
- 🔍 **Check a Job Posting** — paste the whole posting as one block (auto-detects salary/company/location signals) or fill a detailed form.
- 📂 **Batch Check (CSV)** — upload a CSV, score every row, download the results.
- 📊 **Model Performance** — see the experiment comparison and pipeline breakdown in-app.

---

## ☁️ Deploy Your Own Copy

1. Push this folder (including `fake_job_stacking_model.joblib`) to a **public** GitHub repo.
   > ⚠️ If the `.joblib` file is over 100 MB, use [Git LFS](https://git-lfs.com/): `git lfs track "*.joblib"`.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Sign in with GitHub**.
3. Click **New app** → select your repo/branch → set main file to `app.py` → **Deploy**.
4. Streamlit installs `requirements.txt` and gives you a public link like:
   `https://your-app-name.streamlit.app`
5. Any future `git push` auto-redeploys the same link.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Modeling | scikit-learn, imbalanced-learn, XGBoost, Hugging Face Transformers (DistilBERT, EX5) |
| Text vectorization | TF-IDF |
| Deployment | Streamlit, Plotly |
| Data | pandas, NumPy |

---

## ⚠️ Disclaimer

This model is a **decision-support tool**, not a certainty. Predictions are based on statistical patterns learned from historical postings and can be wrong — always use human judgment before acting on a fraud flag.

---

## 📄 License

This project is released under the MIT License — free to use, modify, and share.

---

<div align="center">

Built with 🔷 for **Samsung Innovation Campus (SIC)** — AI Track

</div>
