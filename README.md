# Insurance Claim Settlement — Bias & Performance Analysis Dashboard

A Streamlit dashboard for investigating bias in insurance claim settlement
decisions, and benchmarking classification models (KNN, Decision Tree,
Random Forest, Gradient Boosting) that predict claim outcome.

## What it does

1. **Descriptive analytics** — claim volumes and approval/repudiation rates
   cross-tabbed against team/zone and other policy attributes.
2. **Diagnostic analysis (bias investigation)** — raw approval-rate gaps by
   age, income, and team, *plus* a confounder-controlled logistic regression
   that tests whether each of those still has a statistically significant,
   independent effect on approval odds once claim amount, cause of claim,
   payment mode, and early/non-early status are accounted for. This is the
   part that actually distinguishes "this group looks different" from
   "this group is treated differently for the same kind of claim."
3. **Feature engineering + model training** — cleans the raw fields, reduces
   high-cardinality categories, one-hot encodes, scales for KNN, and trains
   all four classifiers on a stratified train/test split.
4. **Model evaluation** — train vs. test accuracy, precision/recall/F1, ROC
   curves (all four models overlaid, to compare stability), and confusion
   matrices, plus feature importance from the tree-based models.
5. **Findings** — an auto-generated written summary that pulls together the
   diagnostic test results and model performance into a single page.

## Project structure

```
.
├── app.py                      # Main Streamlit app (tabs/navigation)
├── utils/
│   ├── data_processing.py      # CSV loading, cleaning, feature engineering
│   ├── stats_tests.py          # Chi-square, Cramér's V, controlled logistic regression
│   └── modeling.py             # Model training + evaluation metrics
├── requirements.txt
├── .streamlit/config.toml      # Theme
└── .gitignore                  # Excludes *.csv by default (see note below)
```

## Expected data format

A CSV with these columns (matches a life-insurance death-claim register):

`POLICY_NO, PI_GENDER, SUM_ASSURED, ZONE, PAYMENT_MODE, EARLY_NON,
PI_OCCUPATION, MEDICAL_NONMED, PI_STATE, REASON_FOR_CLAIM, PI_AGE,
PI_ANNUAL_INCOME, POLICY_STATUS`

- `ZONE` is treated as the team/branch that handled the claim.
- `POLICY_STATUS` should contain the literal text "Approved" for approved
  claims (e.g. "Approved Death Claim"); anything else is treated as repudiated.
- `SUM_ASSURED` / `PI_ANNUAL_INCOME` can be plain numbers or Indian-formatted
  strings with commas (e.g. `"1,68,300"`).

If your data uses different column names, the easiest path is to rename
your columns to match before uploading — the app validates column presence
on upload and will tell you exactly what's missing.

## ⚠️ Important: don't commit real claims data to a public GitHub repo

This dashboard is designed so you **upload your CSV at runtime** through the
sidebar — it is never required to live in the repo. The `.gitignore` already
excludes `*.csv` for this reason. If your organization's claims data is
confidential (it almost certainly is — it contains names, ages, income, and
medical cause of death), do not add it to git, even in a private repo,
unless you've confirmed that's permitted by your data-handling policy.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`),
and upload your CSV from the sidebar.

## Deploy to Streamlit Community Cloud (free)

1. Push this folder to a new GitHub repository (data file **not** included —
   see warning above).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **"New app"**, select your repo/branch, and set the main file path
   to `app.py`.
4. Click **Deploy**. Once it's live, upload your CSV through the app's
   sidebar each time you open it (it isn't stored on the server between
   sessions).

If your repo is private, Streamlit Community Cloud can still deploy it —
just authorize access when prompted.

## Notes on the modeling approach

- All four models predict the **existing** `POLICY_STATUS` label, so they
  learn whatever pattern produced that label historically — legitimate or
  not. Treat them as a diagnostic lens on the data, not as a recommendation
  engine for real claim decisions.
- Class imbalance (roughly 65–70% approved in typical claim registers) is
  handled by stratified train/test splitting and `class_weight="balanced"`
  for Decision Tree and Random Forest. KNN and Gradient Boosting don't
  support that parameter in scikit-learn, so watch their recall/precision
  trade-off in the evaluation tab.
- High-cardinality fields (team/zone, state, occupation, cause of claim) are
  collapsed to the top N categories + "Other" before one-hot encoding — you
  can adjust N in the sidebar. Lowering it reduces the feature count (good
  for KNN) at the cost of losing team-level granularity.
