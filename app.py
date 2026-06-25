"""
Insurance Claim Settlement — Bias & Performance Analysis Dashboard
====================================================================
Streamlit dashboard to investigate potential bias in claim settlement
decisions (by team/zone, age, income) and benchmark classification models
(KNN, Decision Tree, Random Forest, Gradient Boosting) that predict claim
outcome from claim-related features.

Run locally:    streamlit run app.py
Deploy:         push this repo to GitHub, then deploy on
                share.streamlit.io (Streamlit Community Cloud)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_processing import load_and_clean, engineer_features, validate_columns, REQUIRED_COLUMNS
from utils.stats_tests import association_table, group_rate_table, controlled_logit
from utils.modeling import train_and_evaluate, metrics_dataframe

st.set_page_config(
    page_title="Claim Settlement Bias Analysis",
    page_icon="📋",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Sidebar — data upload & model configuration
# ----------------------------------------------------------------------------
st.sidebar.title("📋 Claim Settlement Analysis")
st.sidebar.caption("Descriptive • Diagnostic • Predictive analytics on claim settlement data")

uploaded_file = st.sidebar.file_uploader(
    "Upload claims CSV",
    type=["csv"],
    help="Expected columns: " + ", ".join(REQUIRED_COLUMNS),
)

if uploaded_file is None:
    st.title("📋 Insurance Claim Settlement — Bias & Performance Dashboard")
    st.info(
        "👈 Upload a claims CSV from the sidebar to get started.\n\n"
        "**Expected columns:** " + ", ".join(REQUIRED_COLUMNS)
    )
    st.markdown(
        """
        This dashboard helps a claim settlement team investigate **whether claim
        outcomes are being driven by legitimate factors (claim amount, cause,
        underwriting type) or by attributes that shouldn't matter (age, income,
        which team/zone handled the file)** — and benchmarks four classification
        models for predicting claim outcome.

        **What it covers**
        1. Descriptive analytics — claim volumes & approval rates across teams/zones
        2. Diagnostic analysis — statistically controlled bias testing (age / income / team)
        3. Feature-engineered classification models — KNN, Decision Tree, Random Forest, Gradient Boosting
        4. Full model evaluation — accuracy, precision/recall/F1, ROC curves, confusion matrices
        5. Auto-generated findings summary
        """
    )
    st.stop()

raw_bytes = uploaded_file.getvalue()
df_preview = pd.read_csv(pd.io.common.BytesIO(raw_bytes), nrows=5)
missing_cols = validate_columns(df_preview)
if missing_cols:
    st.error(f"Uploaded file is missing required columns: {missing_cols}")
    st.stop()

df = load_and_clean(raw_bytes)

st.sidebar.success(f"Loaded {len(df):,} claims")
st.sidebar.metric("Overall Approval Rate", f"{df['STATUS_BIN'].mean()*100:.1f}%")

with st.sidebar.expander("⚙️ Model settings", expanded=False):
    test_size = st.slider("Test set size", 0.15, 0.4, 0.25, 0.05)
    cardinality_top_n = st.slider("Max categories per field before grouping into 'Other'", 5, 20, 12)
    st.markdown("**Hyperparameters**")
    knn_k = st.slider("KNN — n_neighbors", 3, 31, 15, 2)
    dt_depth = st.slider("Decision Tree — max_depth", 2, 15, 7)
    rf_trees = st.slider("Random Forest — n_estimators", 50, 400, 200, 50)
    rf_depth = st.slider("Random Forest — max_depth", 2, 20, 10)
    gb_trees = st.slider("Gradient Boosting — n_estimators", 50, 400, 150, 50)
    gb_depth = st.slider("Gradient Boosting — max_depth", 1, 8, 3)

X, y, feat_meta = engineer_features(df, cardinality_top_n=cardinality_top_n)

model_kwargs = dict(knn_k=knn_k, dt_depth=dt_depth, rf_depth=rf_depth,
                     rf_trees=rf_trees, gb_trees=gb_trees, gb_depth=gb_depth)

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_overview, tab_desc, tab_diag, tab_model, tab_eval, tab_findings = st.tabs(
    ["🏠 Overview", "📊 Descriptive Analytics", "🔍 Diagnostic Analysis (Bias)",
     "🤖 Feature Engineering & Models", "📐 Model Evaluation", "📝 Findings"]
)

# ============================================================================
# TAB 1 — OVERVIEW
# ============================================================================
with tab_overview:
    st.header("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Claims", f"{len(df):,}")
    c2.metric("Approved", f"{df['STATUS_BIN'].sum():,}")
    c3.metric("Repudiated", f"{(df['STATUS_BIN']==0).sum():,}")
    c4.metric("Teams / Zones", df["ZONE"].nunique())

    c1, c2, c3 = st.columns(3)
    c1.metric("Median Age", f"{df['PI_AGE'].median():.0f}")
    c2.metric("Median Sum Assured", f"₹{df['SUM_ASSURED'].median():,.0f}")
    pct_income_disclosed = (df["PI_ANNUAL_INCOME"] > 0).mean() * 100
    c3.metric("Income Disclosed", f"{pct_income_disclosed:.0f}% of records")

    if pct_income_disclosed < 70:
        st.warning(
            f"⚠️ **Data quality note:** {100-pct_income_disclosed:.0f}% of records have "
            "`PI_ANNUAL_INCOME = 0`, which looks like non-disclosure rather than genuine "
            "zero income (it spans every occupation type, including business owners and "
            "managers). These are bucketed separately as 'Not Disclosed' in the income-band "
            "analysis rather than treated as the lowest income group."
        )

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(20), width='stretch')

    st.subheader("Missing Values")
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts):
        st.dataframe(null_counts.rename("Missing Count"), width='stretch')
    else:
        st.success("No missing values in the required columns.")

# ============================================================================
# TAB 2 — DESCRIPTIVE ANALYTICS
# ============================================================================
with tab_desc:
    st.header("Descriptive Analytics")
    st.caption("Claim volumes and approval/repudiation patterns across teams, zones, and other policy attributes.")

    st.subheader("Policy Status by Team / Zone")
    zone_ct = pd.crosstab(df["ZONE"], df["POLICY_STATUS"])
    zone_ct["Total"] = zone_ct.sum(axis=1)
    zone_ct = zone_ct.sort_values("Total", ascending=False)
    st.dataframe(zone_ct, width='stretch')

    zone_rate = group_rate_table(df, "ZONE", "STATUS_BIN", min_n=10)
    fig = px.bar(
        zone_rate, x="ZONE", y="approval_rate", color="reliable (n>=10)",
        hover_data=["n"], title="Approval Rate by Team / Zone (color = sample size ≥ 10)",
        labels={"approval_rate": "Approval Rate", "ZONE": "Team / Zone"},
    )
    fig.add_hline(y=df["STATUS_BIN"].mean(), line_dash="dash", annotation_text="Overall average")
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Status by Gender")
        gct = pd.crosstab(df["PI_GENDER"], df["POLICY_STATUS"], normalize="index") * 100
        fig = px.bar(gct, barmode="group", title="% within Gender", labels={"value": "%"})
        st.plotly_chart(fig, width='stretch')

        st.subheader("Status by Early/Non-Early Claim")
        ect = pd.crosstab(df["EARLY_NON"], df["POLICY_STATUS"], normalize="index") * 100
        fig = px.bar(ect, barmode="group", title="% within Early/Non-Early", labels={"value": "%"})
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Status by Medical / Non-Medical Underwriting")
        mct = pd.crosstab(df["MEDICAL_NONMED"], df["POLICY_STATUS"], normalize="index") * 100
        fig = px.bar(mct, barmode="group", title="% within Underwriting Type", labels={"value": "%"})
        st.plotly_chart(fig, width='stretch')

        st.subheader("Status by Payment Mode")
        pct_df = pd.crosstab(df["PAYMENT_MODE"], df["POLICY_STATUS"], normalize="index") * 100
        fig = px.bar(pct_df, barmode="group", title="% within Payment Mode", labels={"value": "%"})
        st.plotly_chart(fig, width='stretch')

    st.subheader("Top States by Claim Volume")
    state_rate = group_rate_table(df, "PI_STATE", "STATUS_BIN", min_n=10).head(15)
    fig = px.bar(state_rate.sort_values("n", ascending=True), x="n", y="PI_STATE", orientation="h",
                 color="approval_rate", title="Claim Volume by State (color = approval rate)")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Top Causes of Claim (Reason for Claim)")
    reason_counts = df["REASON_FOR_CLAIM"].value_counts().head(15)
    fig = px.bar(reason_counts, orientation="h", title="Top 15 Causes")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
    st.plotly_chart(fig, width='stretch')

# ============================================================================
# TAB 3 — DIAGNOSTIC ANALYSIS (BIAS)
# ============================================================================
with tab_diag:
    st.header("Diagnostic Analysis — Probing for Bias")
    st.caption(
        "Raw approval-rate gaps between groups can be driven by legitimate confounders "
        "(claim amount, cause of death, underwriting type) rather than the group itself. "
        "So below, raw rates are shown first, then a **confounder-controlled logistic "
        "regression** tests whether age, income, or team still matter once those "
        "legitimate factors are accounted for."
    )

    st.subheader("1. Raw Approval Rates")
    c1, c2, c3 = st.columns(3)
    with c1:
        age_rate = df.groupby("AGE_BAND")["STATUS_BIN"].agg(["mean", "count"]).reset_index()
        fig = px.bar(age_rate, x="AGE_BAND", y="mean", text=age_rate["count"],
                     title="Approval Rate by Age Band", labels={"mean": "Approval Rate"})
        fig.add_hline(y=df["STATUS_BIN"].mean(), line_dash="dash")
        st.plotly_chart(fig, width='stretch')
    with c2:
        inc_rate = df.groupby("INCOME_BAND", observed=True)["STATUS_BIN"].agg(["mean", "count"]).reset_index()
        fig = px.bar(inc_rate, x="INCOME_BAND", y="mean", text=inc_rate["count"],
                     title="Approval Rate by Income Band", labels={"mean": "Approval Rate"})
        fig.add_hline(y=df["STATUS_BIN"].mean(), line_dash="dash")
        st.plotly_chart(fig, width='stretch')
    with c3:
        zone_rate2 = group_rate_table(df, "ZONE", "STATUS_BIN", min_n=15)
        zone_rate2 = zone_rate2[zone_rate2["n"] >= 15]
        fig = px.bar(zone_rate2.sort_values("approval_rate"), x="approval_rate", y="ZONE", orientation="h",
                     title="Approval Rate by Team (n≥15)", labels={"approval_rate": "Approval Rate"})
        fig.add_vline(x=df["STATUS_BIN"].mean(), line_dash="dash")
        fig.update_layout(height=450)
        st.plotly_chart(fig, width='stretch')

    st.subheader("2. Association Tests (Chi-Square + Cramér's V)")
    st.caption(
        "Cramér's V is the effect size (0 = no association, closer to 1 = strong). "
        "With ~1,800 rows, even small differences can be 'statistically significant' "
        "(p<0.05) without being practically large — that's why Cramér's V matters as much as the p-value."
    )
    assoc_cols = ["ZONE", "AGE_BAND", "INCOME_BAND", "PI_GENDER", "EARLY_NON",
                  "MEDICAL_NONMED", "PAYMENT_MODE", "PI_STATE"]
    assoc_tbl = association_table(df, "STATUS_BIN", assoc_cols)
    st.dataframe(
        assoc_tbl.style.format({"p_value": "{:.2e}"}),
        width='stretch',
    )

    st.subheader("3. Confounder-Controlled Bias Test (Logistic Regression)")
    st.caption(
        "Each test below controls for claim amount, early/non-early status, medical/"
        "non-medical underwriting, payment mode, and cause of claim — then checks "
        "whether the sensitive attribute *still* shifts the odds of approval."
    )

    control_cats = ["EARLY_NON", "MEDICAL_NONMED", "REASON_FOR_CLAIM", "PAYMENT_MODE"]

    diag_tabs = st.tabs(["Age", "Income", "Team / Zone"])

    with diag_tabs[0]:
        res = controlled_logit(df, "PI_AGE", "numeric", control_cats)
        coef = res["breakdown"].loc["PI_AGE"]
        p = coef["p_value"]
        verdict = "🔴 Statistically significant" if p < 0.05 else "🟢 Not statistically significant"
        st.markdown(f"**Effect of age (per year), controlling for claim amount/type/payment:** "
                    f"odds ratio = `{coef['odds_ratio']:.4f}`, p-value = `{p:.4f}` → {verdict}")
        if p >= 0.05:
            st.info(
                "Age does **not** show an independent effect on approval odds once claim "
                "amount, cause of claim, payment mode, and early/non-early status are "
                "controlled for. The raw age-band gap seen above is likely explained by "
                "those confounders (e.g. older claimants more often have early/non-early "
                "or cause-of-death profiles that correlate with approval)."
            )
        else:
            st.warning("Age shows a statistically significant independent effect — worth deeper review.")

    with diag_tabs[1]:
        res = controlled_logit(df, "PI_ANNUAL_INCOME", "numeric", control_cats)
        key = "PI_ANNUAL_INCOME_LAKH" if "PI_ANNUAL_INCOME_LAKH" in res["breakdown"].index else "PI_ANNUAL_INCOME"
        coef = res["breakdown"].loc[key]
        p = coef["p_value"]
        verdict = "🔴 Statistically significant" if p < 0.05 else "🟢 Not statistically significant"
        st.markdown(f"**Effect of income (per ₹1 lakh), controlling for claim amount/type/payment:** "
                    f"odds ratio = `{coef['odds_ratio']:.4f}`, p-value = `{p:.4f}` → {verdict}")
        if p < 0.05:
            direction = "higher" if coef["odds_ratio"] > 1 else "lower"
            st.warning(
                f"Income has a statistically significant independent effect: claimants with "
                f"{direction} income have {direction} odds of approval, even after controlling "
                "for legitimate claim factors. This is consistent with income-based bias and "
                "warrants investigation into underwriting/settlement guidelines."
            )

    with diag_tabs[2]:
        res = controlled_logit(df, "ZONE", "categorical", control_cats, top_n_categories=cardinality_top_n)
        st.markdown(
            f"**Joint test — does team/zone matter at all**, controlling for claim "
            f"amount/type/payment: Likelihood-ratio χ² = `{res['lr_stat']:.1f}`, "
            f"df = `{res['df_diff']}`, p-value = `{res['lr_pvalue']:.2e}`"
        )
        if res["lr_pvalue"] < 0.05:
            st.error(
                "🔴 **Team/zone has a statistically significant effect on approval odds, "
                "independent of claim amount, cause, payment mode, and early/non-early "
                "status.** This is the strongest evidence of process inconsistency in the "
                "dataset — different teams are settling comparable claims differently."
            )
        st.markdown("**Per-team odds ratio vs. the most common team (reference group):**")
        bd = res["breakdown"].copy()
        bd.index = [i.replace("ZONE_", "") for i in bd.index]
        bd["Significant"] = bd["p_value"] < 0.05
        st.dataframe(
            bd.style.format({"coef": "{:.3f}", "odds_ratio": "{:.2f}", "p_value": "{:.4f}"}),
            width='stretch',
        )
        sig_teams = bd[bd["Significant"]]
        if len(sig_teams):
            higher = sig_teams[sig_teams["odds_ratio"] > 1].index.tolist()
            lower = sig_teams[sig_teams["odds_ratio"] < 1].index.tolist()
            msg = ""
            if higher:
                msg += f"**Approve significantly more readily than reference:** {', '.join(higher)}.\n\n"
            if lower:
                msg += f"**Approve significantly less readily than reference:** {', '.join(lower)}."
            st.markdown(msg)

# ============================================================================
# TAB 4 — FEATURE ENGINEERING & MODELS
# ============================================================================
with tab_model:
    st.header("Feature Engineering & Model Training")

    st.subheader("Feature Engineering Steps Applied")
    st.markdown(
        f"""
        1. **Numeric cleaning** — `SUM_ASSURED` and `PI_ANNUAL_INCOME` parsed from
           Indian-formatted strings (e.g. `"1,68,300"`) into numeric values.
        2. **Missing value handling** — `PI_OCCUPATION` → `'Unknown'`,
           `REASON_FOR_CLAIM` → `'Not Specified'`.
        3. **Target encoding** — `POLICY_STATUS` → binary `STATUS_BIN`
           (1 = Approved, 0 = Repudiated).
        4. **Cardinality reduction** — categorical fields with many rare levels
           (`ZONE`, `PI_STATE`, `PI_OCCUPATION`, `REASON_FOR_CLAIM`) collapsed
           to the top **{cardinality_top_n}** categories + `'Other'`, to keep
           one-hot encoding from exploding into hundreds of sparse columns.
        5. **One-hot encoding** — all categorical features encoded with
           `drop_first=True` to avoid the dummy-variable trap.
        6. **Scaling** — `StandardScaler` applied for KNN only (distance-based);
           tree ensembles use the raw/one-hot features directly.
        7. **Stratified train/test split** — {int((1-test_size)*100)}/{int(test_size*100)}
           split, stratified on the target to preserve the approval/repudiation ratio.

        **Final feature matrix:** `{X.shape[0]:,}` rows × `{X.shape[1]}` columns.
        """
    )

    with st.expander("Preview engineered feature matrix (X)"):
        st.dataframe(X.head(10), width='stretch')

    st.subheader("Train Classification Models")
    st.caption("KNN • Decision Tree • Random Forest • Gradient Boosting — trained on the feature matrix above.")

    if st.button("🚀 Train / Re-train Models", type="primary"):
        with st.spinner("Training KNN, Decision Tree, Random Forest, and Gradient Boosting..."):
            bundle = train_and_evaluate(X, y, test_size=test_size, model_kwargs=model_kwargs)
        st.session_state["model_bundle"] = bundle
        st.session_state["X_columns"] = list(X.columns)
        st.success("Models trained. Open the **Model Evaluation** tab to see results.")

    if "model_bundle" in st.session_state:
        st.info("✅ A trained model bundle is available — see the **Model Evaluation** tab.")
    else:
        st.warning("Click the button above to train the models before viewing the Evaluation tab.")

# ============================================================================
# TAB 5 — MODEL EVALUATION
# ============================================================================
with tab_eval:
    st.header("Model Evaluation")

    if "model_bundle" not in st.session_state:
        st.warning("⚠️ No trained models yet. Go to the **Feature Engineering & Models** tab and click "
                   "'Train / Re-train Models' first.")
        st.stop()

    bundle = st.session_state["model_bundle"]
    results = bundle["results"]

    st.subheader("Accuracy, Precision, Recall, F1, ROC-AUC")
    metrics_df = metrics_dataframe(results)
    st.dataframe(metrics_df.style.format("{:.3f}").background_gradient(cmap="RdYlGn", subset=[
        "Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"
    ]), width='stretch')

    st.caption(
        "**Overfit Gap** = Train Accuracy − Test Accuracy. A large gap (e.g. >0.10) "
        "suggests the model has memorized the training data rather than learning a "
        "generalizable pattern — watch this for Decision Tree / Random Forest in particular."
    )

    st.subheader("Train vs. Test Accuracy")
    acc_long = metrics_df[["Train Accuracy", "Test Accuracy"]].reset_index().melt(
        id_vars="Model", var_name="Split", value_name="Accuracy"
    )
    fig = px.bar(acc_long, x="Model", y="Accuracy", color="Split", barmode="group",
                 title="Train vs Test Accuracy by Model", range_y=[0, 1])
    st.plotly_chart(fig, width='stretch')

    st.subheader("Precision / Recall / F1 Score")
    prf_long = metrics_df[["Precision", "Recall", "F1 Score"]].reset_index().melt(
        id_vars="Model", var_name="Metric", value_name="Score"
    )
    fig = px.bar(prf_long, x="Model", y="Score", color="Metric", barmode="group",
                 title="Precision / Recall / F1 by Model", range_y=[0, 1])
    st.plotly_chart(fig, width='stretch')

    st.subheader("ROC Curves (Model Stability)")
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines",
                                  name=f"{name} (AUC={r['auc']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess",
                              line=dict(dash="dash", color="gray")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                       title="ROC Curve Comparison", height=500)
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "A model whose ROC curve sits consistently above the others — and close to the "
        "top-left corner — separates approved vs. repudiated claims more reliably across "
        "all decision thresholds. AUC close to 0.5 = no better than random guessing."
    )

    st.subheader("Confusion Matrices")
    cols = st.columns(4)
    for col, (name, r) in zip(cols, results.items()):
        with col:
            cm = r["confusion_matrix"]
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                             labels=dict(x="Predicted", y="Actual"),
                             x=["Repudiated", "Approved"], y=["Repudiated", "Approved"],
                             title=name)
            fig.update_layout(height=320, margin=dict(t=40, b=10))
            st.plotly_chart(fig, width='stretch')

    st.subheader("Feature Importance (Tree-Based Models)")
    imp_cols = st.columns(3)
    tree_models = {k: v for k, v in results.items() if v["feature_importance"] is not None}
    for col, (name, r) in zip(imp_cols, tree_models.items()):
        with col:
            top_feat = r["feature_importance"].head(10).sort_values()
            fig = px.bar(x=top_feat.values, y=top_feat.index, orientation="h",
                         title=f"{name} — Top 10 Features")
            fig.update_layout(height=400, margin=dict(t=40))
            st.plotly_chart(fig, width='stretch')
    st.caption(
        "If sensitive attributes (age, income, team dummies) rank highly here, the model "
        "is leaning on them to predict the outcome — which mirrors, rather than corrects, "
        "any bias present in historical settlement decisions. Use this alongside the "
        "Diagnostic Analysis tab, not as a substitute for it."
    )

# ============================================================================
# TAB 6 — FINDINGS
# ============================================================================
with tab_findings:
    st.header("📝 Findings Summary")
    st.caption("Auto-generated from the current dataset and trained models. Re-run after changing settings.")

    overall_rate = df["STATUS_BIN"].mean()
    st.markdown(f"### Dataset\n- **{len(df):,} claims** analyzed, overall approval rate **{overall_rate*100:.1f}%** "
                f"({df['STATUS_BIN'].sum():,} approved / {(df['STATUS_BIN']==0).sum():,} repudiated).")

    pct_income_disclosed = (df["PI_ANNUAL_INCOME"] > 0).mean() * 100
    if pct_income_disclosed < 70:
        st.markdown(
            f"- ⚠️ **Data quality:** only {pct_income_disclosed:.0f}% of records have a non-zero "
            "`PI_ANNUAL_INCOME`. Any income-based finding below is based on the subset where "
            "income was actually disclosed."
        )

    st.markdown("### Bias Diagnostics")
    control_cats = ["EARLY_NON", "MEDICAL_NONMED", "REASON_FOR_CLAIM", "PAYMENT_MODE"]

    age_res = controlled_logit(df, "PI_AGE", "numeric", control_cats)
    age_p = age_res["breakdown"].loc["PI_AGE", "p_value"]

    inc_res = controlled_logit(df, "PI_ANNUAL_INCOME", "numeric", control_cats)
    inc_key = "PI_ANNUAL_INCOME_LAKH" if "PI_ANNUAL_INCOME_LAKH" in inc_res["breakdown"].index else "PI_ANNUAL_INCOME"
    inc_p = inc_res["breakdown"].loc[inc_key, "p_value"]
    inc_or = inc_res["breakdown"].loc[inc_key, "odds_ratio"]

    zone_res = controlled_logit(df, "ZONE", "categorical", control_cats, top_n_categories=cardinality_top_n)

    if age_p < 0.05:
        st.markdown(f"- 🔴 **Age** has a statistically significant independent effect on approval odds (p={age_p:.4f}), even after controlling for claim amount, cause, payment mode, and early/non-early status.")
    else:
        st.markdown(f"- 🟢 **Age** shows no statistically significant independent effect on approval odds once legitimate claim factors are controlled for (p={age_p:.4f}). The raw gap by age band is likely explained by confounders such as cause of claim.")

    if inc_p < 0.05:
        direction = "higher" if inc_or > 1 else "lower"
        st.markdown(f"- 🔴 **Income** has a statistically significant independent effect (p={inc_p:.4f}, odds ratio={inc_or:.3f} per ₹1 lakh) — claimants with {direction} income have {direction} odds of approval, controlling for claim amount/type/payment. This is consistent with income-based disparity.")
    else:
        st.markdown(f"- 🟢 **Income** shows no statistically significant independent effect on approval odds (p={inc_p:.4f}).")

    if zone_res["lr_pvalue"] < 0.05:
        bd = zone_res["breakdown"]
        sig = bd[bd["p_value"] < 0.05].copy()
        sig.index = [i.replace("ZONE_", "") for i in sig.index]
        higher = sig[sig["odds_ratio"] > 1].sort_values("odds_ratio", ascending=False).index.tolist()
        lower = sig[sig["odds_ratio"] < 1].sort_values("odds_ratio").index.tolist()
        st.markdown(
            f"- 🔴 **Team / Zone** has a statistically significant effect on approval odds "
            f"(likelihood-ratio p={zone_res['lr_pvalue']:.2e}), independent of claim amount, "
            "cause, payment mode, and early/non-early status — i.e. **comparable claims are "
            "being settled differently depending on which team handles them.**"
        )
        if higher:
            st.markdown(f"  - Teams approving **significantly more** than the baseline: {', '.join(higher)}")
        if lower:
            st.markdown(f"  - Teams approving **significantly less** than the baseline: {', '.join(lower)}")
    else:
        st.markdown(f"- 🟢 **Team / Zone** shows no statistically significant effect once legitimate claim factors are controlled for (p={zone_res['lr_pvalue']:.2e}).")

    st.markdown("### Model Performance")
    if "model_bundle" in st.session_state:
        results = st.session_state["model_bundle"]["results"]
        metrics_df = metrics_dataframe(results)
        best_f1 = metrics_df["F1 Score"].idxmax()
        best_auc = metrics_df["ROC AUC"].idxmax()
        most_overfit = metrics_df["Overfit Gap (Train-Test Acc)"].idxmax()
        st.markdown(
            f"- Best F1 score: **{best_f1}** ({metrics_df.loc[best_f1, 'F1 Score']:.3f}). "
            f"Best ROC-AUC (most stable across thresholds): **{best_auc}** ({metrics_df.loc[best_auc, 'ROC AUC']:.3f})."
        )
        gap = metrics_df.loc[most_overfit, "Overfit Gap (Train-Test Acc)"]
        if gap > 0.10:
            st.markdown(f"- ⚠️ **{most_overfit}** shows the largest train/test gap ({gap:.3f}) — a sign of overfitting; consider reducing depth/complexity in the sidebar.")

        tree_models = {k: v for k, v in results.items() if v["feature_importance"] is not None}
        sensitive_in_top = []
        for name, r in tree_models.items():
            top5 = r["feature_importance"].head(5).index.tolist()
            flagged = [f for f in top5 if f.startswith("PI_AGE") or f.startswith("PI_ANNUAL_INCOME") or f.startswith("ZONE_R")]
            if flagged:
                sensitive_in_top.append(f"{name} ({', '.join(flagged)})")
        if sensitive_in_top:
            st.markdown(
                "- ⚠️ Sensitive attributes appear in the **top-5 most important features** for: "
                + "; ".join(sensitive_in_top) +
                ". This means the model is partly predicting outcome *from* these attributes — "
                "if used operationally, it would reproduce rather than correct any historical bias."
            )
    else:
        st.info("Train the models in the **Feature Engineering & Models** tab to see performance findings here.")

    st.markdown("### Caveats")
    st.markdown(
        """
        - Statistical association is not proof of *intentional* bias — it identifies
          **process inconsistency** worth investigating operationally (e.g. auditing a
          sample of decisions from outlier teams).
        - The controlled regression only controls for the variables available in this
          dataset. Unmeasured factors (claim documentation quality, investigator
          discretion, fraud indicators) could still explain part of the team-level gap.
        - Classification models here are trained to predict the *existing* labels in the
          data — they learn whatever pattern (legitimate or biased) produced those labels.
          They are a diagnostic tool, not a recommended replacement for human claim review.
        """
    )
