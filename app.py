import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.pipeline import run_experiment, make_demo_data

st.set_page_config(page_title="RegimeGuard | Market Intelligence", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#080d16; --panel:#101827; --panel2:#141f30; --line:#243247; --muted:#8ea0b9; --text:#edf4ff; --cyan:#57d6e8; --green:#49d7a0; --red:#ff6f86; }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.stApp { background: radial-gradient(ellipse at 80% 0%, #13283a 0%, #080d16 45%); color:var(--text); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#0b1220; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--text); }
.block-container { padding-top:1.7rem; padding-bottom:3rem; max-width:1500px; }
h1,h2,h3 { color:var(--text)!important; letter-spacing:-.04em; }
p, label, .stMarkdown { color:#c5d2e5; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#121e2e,#0e1725); border:1px solid #27374b; padding:18px 20px; border-radius:16px; box-shadow:0 8px 28px #0002; }
div[data-testid="stMetricLabel"] { color:#8ea0b9!important; font-size:.78rem; text-transform:uppercase; letter-spacing:.09em; }
div[data-testid="stMetricValue"] { color:#f1f7ff!important; font-family:'DM Mono',monospace; }
div[data-testid="stMetricDelta"] { font-family:'DM Mono',monospace; }
.stButton button, .stDownloadButton button { border-radius:10px; border:1px solid #3c7182; background:linear-gradient(135deg,#2ca9c3,#217d9a); color:white; font-weight:800; min-height:44px; }
.stButton button:hover, .stDownloadButton button:hover { border-color:#74e5f3; color:white; }
div[data-testid="stFileUploader"] { background:#101827; border:1px dashed #3a506a; border-radius:12px; padding:12px; }
div[data-testid="stTabs"] button { color:#a8b8cc; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#57d6e8; }
div[data-testid="stDataFrame"] { border:1px solid #27374b; border-radius:12px; overflow:hidden; }
hr { border-color:#253348; }
.small-muted { color:#8ea0b9; font-size:.83rem; }
.hero { padding:24px 26px; border:1px solid #26384d; border-radius:20px; background:linear-gradient(120deg,#121f30dd,#0c1421ee); margin-bottom:20px; }
.eyebrow { font-family:'DM Mono',monospace; color:#57d6e8; font-size:.75rem; letter-spacing:.16em; text-transform:uppercase; }
.hero-title { font-size:2.35rem; font-weight:800; letter-spacing:-.06em; color:#f2f7ff; margin:.25rem 0; }
.hero-copy { color:#9fb1c8; max-width:800px; }
.status-pill { display:inline-block; border:1px solid #276c5a; background:#10362f; color:#68e4b2; border-radius:999px; padding:6px 11px; font-size:.75rem; font-weight:800; }
.section-head { font-size:1.2rem; font-weight:800; color:#e8f1ff; margin:8px 0 4px 0; }
.panel { background:#101827; border:1px solid #25364b; border-radius:16px; padding:18px; }
div[data-testid="stAlert"] { border-radius:12px; }
</style>
""", unsafe_allow_html=True)

def chart_theme(fig, height=320):
    fig.update_layout(
        height=height, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope", color="#a9bad0", size=11),
        margin=dict(l=10,r=10,t=48,b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(gridcolor="#223047", zerolinecolor="#223047"),
        yaxis=dict(gridcolor="#223047", zerolinecolor="#223047")
    )
    return fig

with st.sidebar:
    st.markdown("## 📈 RegimeGuard")
    st.markdown("<span class='small-muted'>MARKET INTELLIGENCE WORKSPACE</span>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### Experiment controls")
    data_mode = st.radio("Data source", ["Synthetic demo", "Upload CSV"], help="Demo data contains a simulated distribution shift.")
    n_rows = st.slider("Demo sample size", 1500, 12000, 4000, 500, disabled=data_mode != "Synthetic demo")
    horizon = st.slider("Forecast horizon (updates)", 1, 30, 10)
    reference_size = st.slider("Reference window", 100, 1000, 400, 50)
    current_size = st.slider("Detection window", 50, 500, 150, 25)
    alpha = st.select_slider("KS significance level", options=[0.001, 0.005, 0.01, 0.025, 0.05], value=0.01)
    retrain_size = st.slider("Adaptive training window", 100, 1500, 500, 100)
    uploaded = st.file_uploader("Upload order book CSV", type=["csv"], disabled=data_mode != "Upload CSV")
    st.caption("Required: bid_price, ask_price, bid_size, ask_size")
    run = st.button("▶  Run analysis", type="primary", use_container_width=True)
    st.divider()
    st.markdown("<span class='small-muted'>RESEARCH PROTOTYPE · NOT FOR LIVE TRADING</span>", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = None
if "source_label" not in st.session_state:
    st.session_state.source_label = "Synthetic demo"

if run:
    try:
        if data_mode == "Upload CSV":
            if uploaded is None:
                st.warning("Upload a CSV file first.")
                st.stop()
            raw = pd.read_csv(uploaded)
            st.session_state.source_label = uploaded.name
        else:
            raw = make_demo_data(n_rows)
            st.session_state.source_label = "Synthetic demo"
        with st.spinner("Engineering features, detecting shifts and evaluating models…"):
            st.session_state.results = run_experiment(
                raw, horizon=horizon, reference_size=reference_size,
                current_size=current_size, alpha=float(alpha), retrain_size=retrain_size
            )
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

st.markdown("""
<div class="hero">
  <div class="eyebrow">ADAPTIVE MARKET FORECASTING · RESEARCH LAB</div>
  <div class="hero-title">RegimeGuard<span style="color:#57d6e8">.</span></div>
  <div class="hero-copy">Forecast short-horizon order-book movement. Detect when market features drift, then compare an adaptive model against a static baseline — all in one monitoring workspace.</div>
  <div style="margin-top:16px"><span class="status-pill">● SHIFT MONITORING READY</span></div>
</div>
""", unsafe_allow_html=True)

res = st.session_state.results
if res is None:
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("### Start an experiment")
        st.write("Run a reproducible demo or upload your own order-book snapshots. The demo includes an injected regime shift so you can inspect the detection workflow.")
        st.markdown("""
        <div class="panel">
        <div class="section-head">What you'll get</div>
        <p>01 &nbsp; Direction forecast from mid-price movement</p>
        <p>02 &nbsp; Rolling KS tests across order-book features</p>
        <p>03 &nbsp; Static vs. adaptive model evaluation</p>
        <p>04 &nbsp; Shift event log and downloadable predictions</p>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("### Feature pipeline")
        st.markdown("""
        <div class="panel">
        <p>◉ <b>Market snapshot</b><br><span class="small-muted">Best bid/ask and quantities</span></p>
        <p style="color:#57d6e8">↓</p>
        <p>◉ <b>Feature engineering</b><br><span class="small-muted">Spread · imbalance · depth · volatility</span></p>
        <p style="color:#57d6e8">↓</p>
        <p>◉ <b>Shift detector</b><br><span class="small-muted">Rolling two-sample KS tests</span></p>
        <p style="color:#57d6e8">↓</p>
        <p>◉ <b>Adaptive forecasting</b><br><span class="small-muted">Recent-window retraining</span></p>
        </div>
        """, unsafe_allow_html=True)
else:
    pred = res["predictions"].copy()
    metrics = res["metrics"]
    alerts = int(pred["shift_detected"].sum())
    delta = metrics["adaptive_accuracy"] - metrics["static_accuracy"]
    st.markdown(f"<span class='small-muted'>DATA SOURCE: {st.session_state.source_label} &nbsp; / &nbsp; TEST ROWS: {len(pred):,} &nbsp; / &nbsp; FORECAST HORIZON: {horizon} updates</span>", unsafe_allow_html=True)
    st.write("")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Static accuracy", f'{metrics["static_accuracy"]:.1%}')
    m2.metric("Adaptive accuracy", f'{metrics["adaptive_accuracy"]:.1%}', f'{delta:+.1%}')
    m3.metric("Shift alerts", f'{alerts:,}')
    m4.metric("Balanced accuracy", f'{metrics["adaptive_balanced_accuracy"]:.1%}')
    m5.metric("Median latency", f'{metrics["median_inference_ms"]:.2f} ms')
    tabs = st.tabs(["Overview", "Shift monitor", "Model evaluation", "Data explorer", "Methodology"])
    with tabs[0]:
        c1,c2 = st.columns([1.55,1])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred["row"], y=pred["mid_price"], name="Mid-price", mode="lines",
                                     line=dict(color="#57d6e8", width=1.7)))
            shifts = pred[pred["shift_detected"]]
            if len(shifts):
                fig.add_trace(go.Scatter(x=shifts["row"], y=shifts["mid_price"], name="Detected shift",
                                         mode="markers", marker=dict(color="#ff6f86", size=9, symbol="x")))
            fig.update_layout(title="Market mid-price & shift events", xaxis_title="Observation", yaxis_title="Mid-price")
            st.plotly_chart(chart_theme(fig, 370), use_container_width=True)
        with c2:
            counts = pd.DataFrame({"Model":["Static baseline","Adaptive model"],
                                   "Accuracy":[metrics["static_accuracy"],metrics["adaptive_accuracy"]]})
            fig2 = px.bar(counts, x="Model", y="Accuracy", color="Model",
                          color_discrete_sequence=["#60758f","#49d7a0"], range_y=[0,1],
                          title="Out-of-sample accuracy")
            fig2.update_layout(showlegend=False, yaxis_tickformat=".0%")
            st.plotly_chart(chart_theme(fig2,370), use_container_width=True)
        st.markdown("#### Latest predictions")
        st.dataframe(pred.tail(12)[["row","mid_price","target","static_pred","adaptive_pred","shift_detected","ks_min_pvalue"]].sort_values("row", ascending=False),
                     use_container_width=True, hide_index=True)
    with tabs[1]:
        c1,c2 = st.columns([1.4,1])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred["row"], y=pred["ks_min_pvalue"], mode="lines",
                                     name="Minimum KS p-value", line=dict(color="#b49aff",width=1.6)))
            fig.add_hline(y=alpha/5, line_dash="dash", line_color="#ff6f86",
                          annotation_text=f"Corrected threshold {alpha/5:.4f}")
            fig.update_layout(title="Distribution-shift detector", xaxis_title="Observation", yaxis_title="Minimum p-value")
            st.plotly_chart(chart_theme(fig,350), use_container_width=True)
        with c2:
            st.markdown("#### Detection summary")
            st.metric("Alerts generated", alerts)
            st.metric("Corrected per-feature threshold", f"{alpha/5:.4f}")
            st.write("A low p-value indicates that the recent feature window differs from the reference window. Alerts are evidence of input drift, not proof that price predictability changed.")
        st.markdown("#### Shift event log")
        if res["events"].empty:
            st.info("No statistically significant shifts detected with these settings. Try a larger detection window or a different significance level.")
        else:
            st.dataframe(res["events"], use_container_width=True, hide_index=True)
    with tabs[2]:
        c1,c2 = st.columns(2)
        with c1:
            correctness = pd.DataFrame({"Observation":pred["row"],"Static":pred["static_correct"],"Adaptive":pred["adaptive_correct"]})
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=correctness["Observation"],y=correctness["Static"].rolling(80,min_periods=10).mean(),name="Static rolling accuracy",line=dict(color="#8196b2")))
            fig.add_trace(go.Scatter(x=correctness["Observation"],y=correctness["Adaptive"].rolling(80,min_periods=10).mean(),name="Adaptive rolling accuracy",line=dict(color="#49d7a0")))
            fig.update_layout(title="Rolling accuracy (80 observations)",yaxis_tickformat=".0%",yaxis_range=[0,1])
            st.plotly_chart(chart_theme(fig,350),use_container_width=True)
        with c2:
            table = pd.DataFrame([
                {"Metric":"Accuracy","Static":metrics["static_accuracy"],"Adaptive":metrics["adaptive_accuracy"]},
                {"Metric":"Balanced accuracy","Static":metrics["static_balanced_accuracy"],"Adaptive":metrics["adaptive_balanced_accuracy"]},
                {"Metric":"F1 score","Static":metrics["static_f1"],"Adaptive":metrics["adaptive_f1"]},
                {"Metric":"Median inference latency (both predictions)","Static":np.nan,"Adaptive":metrics["median_inference_ms"]},
            ])
            st.markdown("#### Model scorecard")
            st.dataframe(table.style.format({"Static":"{:.3f}","Adaptive":"{:.3f}"}),use_container_width=True,hide_index=True)
            st.caption("Accuracy and F1 depend on the dataset, horizon and class balance. Compare multiple runs before drawing conclusions.")
    with tabs[3]:
        st.markdown("#### Prediction records")
        search = st.text_input("Filter by row number (optional)", placeholder="e.g. 2500")
        shown = pred
        if search.strip().isdigit():
            shown = pred[pred["row"].astype(str).str.contains(search.strip(), regex=False)]
        st.dataframe(shown, use_container_width=True, hide_index=True, height=400)
        st.download_button("⬇ Download prediction CSV", pred.to_csv(index=False), "regimeguard_predictions.csv", "text/csv")
        st.download_button("⬇ Download shift event CSV", res["events"].to_csv(index=False), "regimeguard_shift_events.csv", "text/csv")
    with tabs[4]:
        st.markdown("#### How the system works")
        st.markdown("""
        **1. Feature engineering** — derives mid-price, spread, order-book imbalance, total depth, returns and rolling volatility.

        **2. Forecast target** — predicts whether the mid-price after the configured number of updates is above its current value.

        **3. Shift detection** — compares recent and reference feature windows with a two-sample Kolmogorov–Smirnov test. A Bonferroni correction is applied across the monitored features.

        **4. Adaptation** — when a shift is flagged, retrains the adaptive model on a recent window of observations with labels assumed to be available by that point.

        **5. Evaluation** — compares static and adaptive models on sequential test observations.
        """)
        st.warning("Research caveat: input-distribution shift is not the same as concept drift. The synthetic demo validates that the pipeline runs; it does not establish real-market profitability or performance.")
        st.markdown("**Recommended next steps:** validate on chronological real LOB data, measure detection delay and false alarms, add probability calibration, and benchmark p50/p95/p99 latency under a defined hardware budget.")

st.divider()
st.markdown("<div class='small-muted'>REGIMEGUARD · ADAPTIVE ORDER BOOK FORECASTING · Educational research prototype · No financial advice</div>", unsafe_allow_html=True)
