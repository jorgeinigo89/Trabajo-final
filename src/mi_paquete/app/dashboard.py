"""Streamlit dashboard for Bank Marketing ML project."""

import sys
from pathlib import Path

# Make the package importable when running directly with `streamlit run`
ROOT = Path(__file__).resolve().parents[3]  # app/mi_paquete/src/<project root>
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

import base64

from mi_paquete.data.loader import load_bank_data, basic_info
from mi_paquete.features.preprocessing import encode_features, get_X_y
from mi_paquete.models.train import split_data, train_random_forest, feature_importances
from mi_paquete.evaluation.metrics import evaluate

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Marketing Dashboard",
    page_icon="🏦",
    layout="wide",
)

# ── Custom font (Bookerly) ────────────────────────────────────────────────────
_font_path = ROOT / "fonts" / "Bookerly.ttf"
if _font_path.exists():
    _font_b64 = base64.b64encode(_font_path.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        @font-face {{
            font-family: 'Bookerly';
            src: url('data:font/ttf;base64,{_font_b64}') format('truetype');
        }}
        html, body, [class*="css"], .stMarkdown, .stText,
        .stDataFrame, .stTable, .stSelectbox, .stSlider,
        .stRadio, .stButton, .stMetric, h1, h2, h3, h4, h5, h6, p, li, td, th {{
            font-family: 'Bookerly', Georgia, serif !important;
            font-size: 14px !important;
        }}
        h1 {{ font-size: 1.8rem !important; }}
        h2 {{ font-size: 1.4rem !important; }}
        h3 {{ font-size: 1.2rem !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ── Developer info ────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Developer:** Jorge Inigo")
st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-jorgeinigo89-181717?logo=github)](https://github.com/jorgeinigo89)")
st.sidebar.markdown("**Lecturer:** Noe Hernandez (ITAM)")
st.sidebar.markdown("Module I — Professional Certificate in AI & LLMs in Financial Markets (ITAM)")
st.sidebar.markdown("**Version:** 1.0.0")
st.sidebar.markdown("**Date:** 2026-05-06")

# ── Load & cache data / model ─────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_bank_data()


@st.cache_resource
def get_model(n_estimators, max_depth):
    df = get_data()
    df_enc, _ = encode_features(df)
    X, y = get_X_y(df_enc)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_random_forest(X_train, y_train, n_estimators=n_estimators, max_depth=max_depth)
    metrics = evaluate(model, X_test, y_test)
    fi = feature_importances(model, list(X.columns))
    return model, metrics, fi, X_test, y_test


# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.title("Model settings")
n_estimators = st.sidebar.slider("n_estimators", 50, 300, 100, step=50)
max_depth = st.sidebar.slider("max_depth", 3, 20, 10)

st.sidebar.markdown("---")
st.sidebar.info("Data: UCI Bank Marketing dataset  \nTarget: `y` (term deposit subscription)")

# ── Load everything ───────────────────────────────────────────────────────────
df = get_data()
model, metrics, fi_df, X_test, y_test = get_model(n_estimators, max_depth)
info = basic_info(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Bank Marketing — ML Dashboard")
st.markdown(
    """
    This dashboard analyses the **UCI Bank Marketing dataset**, which records the outcomes of
    direct phone-call marketing campaigns run by a Portuguese bank between **May 2008 and
    November 2010**.  
    The goal is to predict whether a client will subscribe to a **term deposit** (`y = yes/no`).

    **Dataset highlights**
    - **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
    - **Campaigns:** outbound calls; some clients were contacted more than once
    - **Features:** client demographics (age, job, marital, education), financial status
      (default, balance, housing loan, personal loan), contact details (type, day, month,
      duration), campaign statistics (contacts, days since last contact, previous outcome)
    - **Target:** `y` — did the client subscribe to a term deposit?
    - **Class imbalance:** ~88 % "no" vs ~12 % "yes"

    The **Random Forest** model below is trained on 80 % of the data.
    Use the sidebar sliders to tune `n_estimators` and `max_depth` interactively.

    ---
    > **Citation:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
    """
)

# ── KPI row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{info['shape'][0]:,}")
col2.metric("Features", info['shape'][1] - 1)
col3.metric("Accuracy", f"{metrics['accuracy']:.2%}")
col4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_eda, tab_model, tab_data = st.tabs(["EDA", "Model Results", "Raw Data"])

# ─── EDA tab ──────────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown(
        """
        ### About the features
        | Feature | Type | Description |
        |---|---|---|
        | `age` | numeric | Client age |
        | `job` | categorical | Type of job |
        | `marital` | categorical | Marital status |
        | `education` | categorical | Education level |
        | `default` | binary | Has credit in default? |
        | `balance` | numeric | Average yearly balance (€) |
        | `housing` | binary | Has housing loan? |
        | `loan` | binary | Has personal loan? |
        | `contact` | categorical | Contact communication type |
        | `day` / `month` | date | Last contact date |
        | `duration` | numeric | Last call duration (seconds) |
        | `campaign` | numeric | Contacts during this campaign |
        | `pdays` | numeric | Days since last contact (−1 = never) |
        | `previous` | numeric | Contacts before this campaign |
        | `poutcome` | categorical | Outcome of previous campaign |
        | **`y`** | **target** | **Term deposit subscribed?** |
        """
    )
    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Target distribution")
        target_counts = df["y"].value_counts().reset_index()
        target_counts.columns = ["y", "count"]
        fig_pie = px.pie(target_counts, names="y", values="count",
                         color_discrete_sequence=["#636EFA", "#EF553B"])
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Age distribution by outcome")
        fig_age = px.histogram(df, x="age", color="y", barmode="overlay",
                               color_discrete_sequence=["#636EFA", "#EF553B"],
                               nbins=40)
        st.plotly_chart(fig_age, use_container_width=True)

    st.subheader("Subscription rate by job")
    job_rate = (
        df.groupby("job")["y"]
        .apply(lambda s: (s == "yes").mean())
        .reset_index()
        .rename(columns={"y": "subscription_rate"})
        .sort_values("subscription_rate", ascending=False)
    )
    fig_job = px.bar(job_rate, x="job", y="subscription_rate",
                     color="subscription_rate", color_continuous_scale="Blues",
                     labels={"subscription_rate": "Subscription rate"})
    st.plotly_chart(fig_job, use_container_width=True)

    st.subheader("Call duration vs. subscription")
    fig_dur = px.box(df, x="y", y="duration", color="y",
                     color_discrete_sequence=["#636EFA", "#EF553B"],
                     labels={"duration": "Call duration (s)", "y": "Subscribed"},
                     points="outliers")
    st.plotly_chart(fig_dur, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Subscription rate by education")
        edu_rate = (
            df.groupby("education")["y"]
            .apply(lambda s: (s == "yes").mean())
            .reset_index()
            .rename(columns={"y": "rate"})
            .sort_values("rate", ascending=False)
        )
        fig_edu = px.bar(edu_rate, x="education", y="rate",
                         color="rate", color_continuous_scale="Purples",
                         labels={"rate": "Subscription rate"})
        st.plotly_chart(fig_edu, use_container_width=True)

    with c4:
        st.subheader("Subscription rate by marital status")
        mar_rate = (
            df.groupby("marital")["y"]
            .apply(lambda s: (s == "yes").mean())
            .reset_index()
            .rename(columns={"y": "rate"})
        )
        fig_mar = px.bar(mar_rate, x="marital", y="rate",
                         color="rate", color_continuous_scale="Oranges",
                         labels={"rate": "Subscription rate"})
        st.plotly_chart(fig_mar, use_container_width=True)

    st.subheader("Numeric feature correlations")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[num_cols].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

# ─── Model tab ────────────────────────────────────────────────────────────────
with tab_model:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Confusion matrix")
        cm = metrics["confusion_matrix"]
        fig_cm = ff.create_annotated_heatmap(
            z=cm.tolist(),
            x=["Pred: no", "Pred: yes"],
            y=["True: no", "True: yes"],
            colorscale="Blues",
        )
        fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig_cm, use_container_width=True)

    with c2:
        st.subheader("Top 10 feature importances")
        fig_fi = px.bar(fi_df.head(10), x="importance", y="feature",
                        orientation="h", color="importance",
                        color_continuous_scale="Teal")
        fig_fi.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_fi, use_container_width=True)

    st.subheader("Classification report")
    st.code(metrics["classification_report"], language="text")

    st.subheader("Test set predictions vs. real values")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    predictions_df = X_test.copy().reset_index(drop=True)
    predictions_df.insert(0, "real", pd.Series(y_test.values, name="real").map({0: "no", 1: "yes"}))
    predictions_df.insert(1, "predicted", pd.Series(y_pred).map({0: "no", 1: "yes"}))
    predictions_df.insert(2, "prob_yes", y_proba.round(3))
    predictions_df.insert(3, "correct", predictions_df["real"] == predictions_df["predicted"])

    filter_opt = st.radio(
        "Show", ["All", "Correct only", "Incorrect only"], horizontal=True
    )
    if filter_opt == "Correct only":
        predictions_df = predictions_df[predictions_df["correct"]]
    elif filter_opt == "Incorrect only":
        predictions_df = predictions_df[~predictions_df["correct"]]

    st.dataframe(
        predictions_df.style.map(
            lambda v: "background-color: #d4edda" if v is True else
                      ("background-color: #f8d7da" if v is False else ""),
            subset=["correct"],
        ),
        use_container_width=True,
        height=450,
    )
    st.caption(f"{len(predictions_df):,} rows shown")

# ─── Raw data tab ─────────────────────────────────────────────────────────────
with tab_data:
    st.markdown(
        """
        Full dataset as loaded from `bank-full.csv`.  
        Use the column headers to sort, and the search box to filter rows.
        """
    )

    # Summary stats expander
    with st.expander("📈 Descriptive statistics"):
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    # Missing values summary
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.success("No missing values found in the dataset.")
    else:
        st.warning("Missing values detected:")
        st.dataframe(missing[missing > 0].rename("missing_count"), use_container_width=True)

    st.subheader("Dataset preview")
    st.dataframe(df, use_container_width=True, height=400)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "bank-full.csv", "text/csv")
