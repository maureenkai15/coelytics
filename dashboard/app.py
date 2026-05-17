import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pickle
from pathlib import Path
import sys
sys.path.append(".")

st.set_page_config(
    page_title="COElytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1400px; }
section[data-testid="stSidebar"] { background: #0d0d14 !important; border-right: 1px solid #1e1e2e; }
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
.kpi-wrap { background: #13131f; border: 1px solid #1e1e30; border-radius: 14px; padding: 1.25rem 1rem 1rem; text-align: center; position: relative; overflow: hidden; }
.kpi-cat  { font-size: 0.65rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px; }
.kpi-val  { font-size: 1.85rem; font-weight: 700; letter-spacing: -1px; line-height: 1; }
.kpi-meta { font-size: 0.72rem; color: #475569; margin-top: 6px; line-height: 1.6; }
.kpi-signal { font-size: 0.7rem; font-weight: 600; padding: 2px 8px; border-radius: 99px; display: inline-block; margin-top: 6px; }
.page-title { font-size: 1.6rem; font-weight: 700; color: #f1f5f9; margin-bottom: 2px; }
.page-sub   { font-size: 0.82rem; color: #475569; margin-bottom: 1.5rem; }
.section-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #475569; margin: 1.5rem 0 0.6rem; }
.insight { background: #0f172a; border-left: 3px solid #6366f1; border-radius: 0 8px 8px 0; padding: 0.75rem 1rem; font-size: 0.82rem; color: #94a3b8; line-height: 1.6; margin-top: 0.75rem; }
.warn    { background: #1a1200; border-left: 3px solid #d97706; border-radius: 0 8px 8px 0; padding: 0.75rem 1rem; font-size: 0.82rem; color: #fcd34d; line-height: 1.6; margin-top: 0.75rem; }
.good    { background: #052e16; border-left: 3px solid #16a34a; border-radius: 0 8px 8px 0; padding: 0.75rem 1rem; font-size: 0.82rem; color: #86efac; line-height: 1.6; margin-top: 0.75rem; }
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 0.5rem; }
.stat-cell { background: #13131f; border: 1px solid #1e1e30; border-radius: 10px; padding: 0.85rem 1rem; }
.stat-cell-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.07em; color: #475569; margin-bottom: 4px; }
.stat-cell-val { font-size: 1.2rem; font-weight: 600; color: #e2e8f0; }
div[data-testid="stMetric"] { background: #13131f !important; border: 1px solid #1e1e30 !important; border-radius: 12px !important; padding: 1rem 1.2rem !important; }
div[data-testid="stMetricValue"] { font-size: 1.35rem !important; font-weight: 700 !important; color: #f1f5f9 !important; }
div[data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: #475569 !important; }
.stApp, .main { background-color: #0a0a0f !important; }
div[data-testid="stMetricValue"] { font-size: 1.35rem !important; font-weight: 700 !important; color: #f1f5f9 !important; }
div[data-testid="stMetricLabel"] { font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.07em; color: #475569 !important; }
.stRadio > label { display: none; }
.main .block-container { background: #0a0a0f !important; }
header[data-testid="stHeader"] { background: #0a0a0f !important; }
.stApp { background: #0a0a0f !important; }
h1, h2 { color: #f1f5f9 !important; font-weight: 700 !important; }
thead tr th { background: #13131f !important; color: #475569 !important; font-size: 0.72rem !important; text-transform: uppercase !important; }
tbody tr td { font-size: 0.82rem !important; }
div[data-baseweb="radio"] label { border-radius: 8px !important; padding: 5px 8px !important; transition: background 0.15s; }
div[data-baseweb="radio"] label:hover { background: #13131f !important; }
.nav-group { font-size: 0.62rem; color: #334155; text-transform: uppercase; letter-spacing: 0.1em; margin: 1rem 0 0.3rem; padding-left: 2px; }
</style>
""", unsafe_allow_html=True)

API_BASE = "http://localhost:8000/api/coe"
MODEL_DIR = Path("ml_models")

CATS = {
    "Category A": {"color": "#818cf8", "label": "Cat A", "desc": "Small cars up to 1600cc - Mass market"},
    "Category B": {"color": "#f472b6", "label": "Cat B", "desc": "Large cars above 1600cc - Premium"},
    "Category C": {"color": "#34d399", "label": "Cat C", "desc": "Goods vehicles and buses"},
    "Category D": {"color": "#fb923c", "label": "Cat D", "desc": "Motorcycles"},
    "Category E": {"color": "#a78bfa", "label": "Cat E", "desc": "Open category - All except motorcycles"},
}
COLOR_MAP = {k: v["color"] for k, v in CATS.items()}

PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    template="plotly_dark",
    margin=dict(l=0, r=0, t=24, b=0),
    font=dict(family="Inter", size=12, color="#64748b"),
)

from backend.services.data_loader import get_latest, get_history, get_stats

@st.cache_data(ttl=300)
def get_latest():
    from backend.services.data_loader import get_latest as _get_latest
    return _get_latest()

@st.cache_data(ttl=300)
def get_history(category=None, start_year=None, end_year=None):
    from backend.services.data_loader import get_history as _get_history
    return _get_history(category=category, start_year=start_year, end_year=end_year)

@st.cache_data(ttl=300)
def get_stats():
    from backend.services.data_loader import get_stats as _get_stats
    return _get_stats()

@st.cache_data(ttl=60)
def get_forecast(category, months):
    from backend.models.predictor import predict_future
    return predict_future(category, months_ahead=months)

@st.cache_data(ttl=60)
def get_history_local(category):
    from backend.models.predictor import load_and_prepare
    return load_and_prepare(category)[["month", "premium"]].tail(24)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 1rem; border-bottom:1px solid #1e1e30; margin-bottom:0.5rem;'>
        <div style='font-size:1.3rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.5px;'>COElytics</div>
        <div style='font-size:0.68rem; color:#334155; margin-top:2px; letter-spacing:0.06em; text-transform:uppercase;'>Singapore Vehicle Intelligence</div>
    </div>
    <div class='nav-group'>Discover</div>
    """, unsafe_allow_html=True)

    page = st.radio("nav", [
        "Market Overview",
        "Price Trends",
        "Category Analysis",
        "ML Price Forecast",
        "Bid Timing Advisor",
        "Affordability Calculator",
        "Total Cost of Ownership",
        "Renew vs Scrap",
    ])

    st.markdown(f"""
    <div style='border-top:1px solid #1e1e30; margin-top:1.5rem; padding-top:1rem;'>
        <div style='font-size:0.68rem; color:#334155;'>Last updated</div>
        <div style='font-size:0.78rem; color:#475569; margin-top:2px;'>{datetime.now().strftime('%d %b %Y, %H:%M')}</div>
        <div style='font-size:0.68rem; color:#334155; margin-top:8px;'>Source</div>
        <div style='font-size:0.78rem; color:#475569; margin-top:2px;'>LTA via data.gov.sg</div>
    </div>
    """, unsafe_allow_html=True)

# PAGE 1 - Market Overview
if page == "Market Overview":
    st.markdown('<div class="page-title">Market Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Live COE premiums across all vehicle categories - Singapore LTA data</div>', unsafe_allow_html=True)

    latest = get_latest()
    df_all = get_history()

    if latest:
        cols = st.columns(5, gap="small")
        for i, item in enumerate(latest):
            cat = item["category"]
            c = CATS[cat]
            quota = item.get("quota") or 1
            bids  = item.get("bids_received") or 0
            ratio = bids / quota
            if ratio > 1.5:   sig, sc, sb = "Hot",  "#ef4444", "#2d0a0a"
            elif ratio > 1.1: sig, sc, sb = "Warm", "#f59e0b", "#1c1200"
            else:             sig, sc, sb = "Cool", "#22c55e", "#052e16"
            with cols[i]:
                st.markdown(f"""
                <div class="kpi-wrap" style="border-color:{c['color']}22;">
                    <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{c['color']};border-radius:14px 14px 0 0;"></div>
                    <div class="kpi-cat" style="color:{c['color']};">{c['label']}</div>
                    <div class="kpi-val" style="color:{c['color']};">${item['premium']:,.0f}</div>
                    <div class="kpi-meta">Quota {quota:,} - {ratio:.1f}x demand</div>
                    <div class="kpi-signal" style="color:{sc};background:{sb};">{sig}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">16-year premium history</div>', unsafe_allow_html=True)
    if not df_all.empty:
        fig = px.line(df_all, x="month", y="premium", color="category",
                      color_discrete_map=COLOR_MAP,
                      labels={"premium": "Premium (SGD)", "month": "", "category": ""})
        fig.update_layout(**PLOT, height=360,
                          yaxis=dict(tickprefix="$", tickformat=",", gridcolor="#13131f", zerolinecolor="#1e1e30"),
                          xaxis=dict(gridcolor="#13131f"),
                          legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"))
        fig.update_traces(line=dict(width=1.6))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        st.markdown('<div class="section-label">Demand pressure - bids vs quota</div>', unsafe_allow_html=True)
        if latest:
            rows = [{"cat": it["category"], "ratio": round((it.get("bids_received") or 0) / (it.get("quota") or 1), 2)} for it in latest]
            df_d = pd.DataFrame(rows)
            fig2 = go.Figure(go.Bar(
                x=df_d["cat"], y=df_d["ratio"],
                marker_color=[COLOR_MAP[c] for c in df_d["cat"]],
                marker_line_width=0,
                text=df_d["ratio"].apply(lambda x: f"{x:.1f}x"),
                textposition="outside", textfont=dict(size=11),
            ))
            fig2.add_hline(y=1.0, line_dash="dot", line_color="#334155")
            fig2.update_layout(**PLOT, height=260,
                               yaxis=dict(gridcolor="#13131f", showticklabels=False),
                               xaxis=dict(tickfont=dict(size=11)))
            st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('<div class="section-label">All-time statistics</div>', unsafe_allow_html=True)
        stats = get_stats()
        if stats:
            for s in stats:
                cat = s["category"]
                col = COLOR_MAP.get(cat, "#818cf8")
                st.markdown(f"""
                <div style="display:flex;align-items:center;justify-content:space-between;
                     padding:10px 14px;background:#13131f;border-radius:10px;
                     border:1px solid #1e1e30;margin-bottom:6px;">
                    <span style="font-size:0.8rem;font-weight:600;color:{col};">{cat}</span>
                    <span style="font-size:0.75rem;color:#475569;">Low <b style="color:#94a3b8">${s['min_premium']:,.0f}</b></span>
                    <span style="font-size:0.75rem;color:#475569;">Avg <b style="color:#94a3b8">${s['avg_premium']:,.0f}</b></span>
                    <span style="font-size:0.75rem;color:#475569;">High <b style="color:#94a3b8">${s['max_premium']:,.0f}</b></span>
                </div>
                """, unsafe_allow_html=True)

# PAGE 2 - Price Trends
elif page == "Price Trends":
    st.markdown('<div class="page-title">Price Trends</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Filter and compare COE premiums across time and categories</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 1], gap="small")
    with c1:
        sel = st.multiselect("Categories", list(CATS.keys()), default=["Category A", "Category B"])
    with c2:
        sy = st.selectbox("From year", list(range(2010, 2027)), index=0)
    with c3:
        ey = st.selectbox("To year", list(range(2010, 2027)), index=16)

    df = get_history(start_year=sy, end_year=ey)
    if not df.empty and sel:
        df_f = df[df["category"].isin(sel)]
        fig = go.Figure()
        for cat in sel:
            d = df_f[df_f["category"] == cat]
            fig.add_trace(go.Scatter(x=d["month"], y=d["premium"], name=cat,
                line=dict(color=COLOR_MAP[cat], width=2),
                hovertemplate=f"<b>{cat}</b> %{{x|%b %Y}}: $%{{y:,.0f}}<extra></extra>"))
        fig.update_layout(**PLOT, height=420, hovermode="x unified",
                          yaxis=dict(tickprefix="$", tickformat=",", gridcolor="#13131f", zerolinecolor="#1e1e30"),
                          xaxis=dict(gridcolor="#13131f"),
                          legend=dict(orientation="h", y=1.06, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-label">Annual averages</div>', unsafe_allow_html=True)
        df_f = df_f.copy()
        df_f["year"] = df_f["month"].dt.year
        pivot = df_f.groupby(["year", "category"])["premium"].mean().round(0).unstack("category")
        pct = pivot.pct_change() * 100
        pivot.index = pivot.index.astype(str)
        pct.index = pct.index.astype(str)
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Average premium (SGD)")
            st.dataframe(pivot.style.format("${:,.0f}").background_gradient(cmap="Blues", axis=None), use_container_width=True)
        with col_b:
            st.caption("Year-on-year change (%)")
            st.dataframe(pct.style.format("{:+.1f}%").background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)

# PAGE 3 - Category Analysis
elif page == "Category Analysis":
    st.markdown('<div class="page-title">Category Analysis</div>', unsafe_allow_html=True)

    sel = st.selectbox("", list(CATS.keys()), label_visibility="collapsed")
    c = CATS[sel]
    st.markdown(f'<div class="page-sub">{c["desc"]}</div>', unsafe_allow_html=True)

    df = get_history(category=sel)
    if not df.empty:
        lp   = df["premium"].iloc[-1]
        pp   = df["premium"].iloc[-2]
        avg  = df["premium"].mean()
        hi   = df["premium"].max()
        lo   = df["premium"].min()
        hi_d = df.loc[df["premium"].idxmax(), "month"].strftime("%b %Y")
        lo_d = df.loc[df["premium"].idxmin(), "month"].strftime("%b %Y")
        pct  = (df["premium"] <= lp).mean() * 100

        cols = st.columns(4, gap="small")
        cols[0].metric("Latest", f"${lp:,.0f}", f"${lp - pp:+,.0f}")
        cols[1].metric(f"Peak - {hi_d}", f"${hi:,.0f}")
        cols[2].metric(f"Trough - {lo_d}", f"${lo:,.0f}")
        cols[3].metric("16yr Average", f"${avg:,.0f}")

        st.markdown('<div class="section-label">Premium history</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["month"], y=df["premium"],
            fill="tozeroy", fillcolor="rgba(129,140,248,0.07)",
            line=dict(color=c["color"], width=2),
            hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra></extra>"))
        fig.add_hline(y=avg, line_dash="dot", line_color="#334155",
                      annotation_text=f"Avg ${avg:,.0f}", annotation_font_color="#475569")
        fig.update_layout(**PLOT, height=320,
                          yaxis=dict(tickprefix="$", tickformat=",", gridcolor="#13131f", zerolinecolor="#1e1e30"),
                          xaxis=dict(gridcolor="#13131f"))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown('<div class="section-label">Quota vs bids</div>', unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df["month"], y=df["bids_received"], name="Bids",
                marker_color="#f472b6", marker_line_width=0, opacity=0.55))
            fig2.add_trace(go.Bar(x=df["month"], y=df["quota"], name="Quota",
                marker_color=c["color"], marker_line_width=0, opacity=0.85))
            fig2.update_layout(**PLOT, barmode="overlay", height=260,
                               yaxis=dict(gridcolor="#13131f"), xaxis=dict(gridcolor="#13131f"),
                               legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            st.markdown('<div class="section-label">Premium distribution</div>', unsafe_allow_html=True)
            fig3 = go.Figure(go.Histogram(x=df["premium"], nbinsx=28,
                marker_color=c["color"], marker_line_width=0, opacity=0.8))
            fig3.add_vline(x=lp, line_dash="dot", line_color="#f1f5f9",
                           annotation_text="Now", annotation_font_color="#94a3b8")
            fig3.update_layout(**PLOT, height=260,
                               xaxis=dict(tickprefix="$", tickformat=",", gridcolor="#13131f"),
                               yaxis=dict(gridcolor="#13131f", title=""))
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown(f'<div class="section-label">Historical percentile - current premium is higher than {pct:.0f}% of all past premiums</div>', unsafe_allow_html=True)
        st.progress(pct / 100)

# PAGE 4 - ML Price Forecast
elif page == "ML Price Forecast":
    st.markdown('<div class="page-title">ML Price Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">XGBoost model trained on 16 years of Singapore COE data</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        sel_cat    = st.selectbox("Category to forecast", list(CATS.keys()))
        months_out = st.slider("Months ahead", 1, 12, 6)
        cat_slug   = sel_cat.replace(" ", "_").lower()
        model_path = MODEL_DIR / f"xgb_{cat_slug}.pkl"

        if model_path.exists():
            with open(model_path, "rb") as f:
                meta = pickle.load(f)
            c = CATS[sel_cat]
            st.markdown(f"""
            <div class="kpi-wrap" style="margin-top:1rem;border-color:{c['color']}22;">
                <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{c['color']};border-radius:14px 14px 0 0;"></div>
                <div class="kpi-cat" style="color:{c['color']};">Model performance</div>
                <div class="stat-grid" style="margin-top:0.75rem;">
                    <div class="stat-cell"><div class="stat-cell-label">MAE</div><div class="stat-cell-val">${meta['mae']:,.0f}</div></div>
                    <div class="stat-cell"><div class="stat-cell-label">MAPE</div><div class="stat-cell-val">{meta['mape']:.1f}%</div></div>
                    <div class="stat-cell"><div class="stat-cell-label">RMSE</div><div class="stat-cell-val">${meta['rmse']:,.0f}</div></div>
                    <div class="stat-cell"><div class="stat-cell-label">Last actual</div><div class="stat-cell-val">${meta['last_premium']:,.0f}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if meta['mape'] < 5:
                st.markdown('<div class="good">Excellent accuracy - MAPE under 5%.</div>', unsafe_allow_html=True)
            elif meta['mape'] < 10:
                st.markdown('<div class="insight">Good accuracy - MAPE under 10%.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warn">Moderate accuracy. Use as directional guidance only.</div>', unsafe_allow_html=True)

    with c2:
        if model_path.exists():
            with st.spinner("Generating forecast..."):
                fc_df = get_forecast(sel_cat, months_out)
                hist  = get_history_local(sel_cat)

            color = CATS[sel_cat]["color"]
            fig   = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist["month"], y=hist["premium"],
                name="Historical", line=dict(color=color, width=2),
                hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=fc_df["month"], y=fc_df["predicted_premium"],
                name="Forecast", line=dict(color="#fbbf24", width=2.5, dash="dot"),
                mode="lines+markers", marker=dict(size=7, color="#fbbf24"),
                hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra></extra>",
            ))
            fig.add_vline(x=hist["month"].iloc[-1].timestamp() * 1000,
                          line_dash="dot", line_color="#334155",
                          annotation_text="Today", annotation_font_color="#475569")
            fig.update_layout(**PLOT, height=380, hovermode="x unified",
                              yaxis=dict(tickprefix="$", tickformat=",", gridcolor="#13131f", zerolinecolor="#1e1e30"),
                              xaxis=dict(gridcolor="#13131f"),
                              legend=dict(orientation="h", y=1.06, bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-label">Forecast table</div>', unsafe_allow_html=True)
            fc_display = fc_df.copy()
            fc_display["month"] = fc_display["month"].dt.strftime("%b %Y")
            fc_display["predicted_premium"] = fc_display["predicted_premium"].apply(lambda x: f"${x:,.0f}")
            fc_display = fc_display[["month", "predicted_premium"]].rename(
                columns={"month": "Month", "predicted_premium": "Predicted Premium"})
            st.dataframe(fc_display, use_container_width=True, hide_index=True)
            st.markdown('<div class="insight">Forecasts are based on historical patterns only. COE premiums are affected by government policy and quota changes not captured in this model. Use as directional guidance only.</div>', unsafe_allow_html=True)
        else:
            st.warning("No model found. Run python3 backend/models/predictor.py first.")

# PAGE 5 - Bid Timing Advisor
elif page == "Bid Timing Advisor":
    st.markdown('<div class="page-title">Bid Timing Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Historical trend analysis to help you decide when to bid</div>', unsafe_allow_html=True)

    sel = st.selectbox("", list(CATS.keys()), label_visibility="collapsed")
    c   = CATS[sel]
    df  = get_history(category=sel)

    if not df.empty and len(df) > 12:
        df = df.sort_values("month").reset_index(drop=True)
        df["ma3"]  = df["premium"].rolling(3).mean()
        df["ma12"] = df["premium"].rolling(12).mean()

        lp    = df["premium"].iloc[-1]
        ma3   = df["ma3"].iloc[-1]
        ma12  = df["ma12"].iloc[-1]
        pct   = (df["premium"] <= lp).mean() * 100
        trend = "rising" if ma3 > ma12 else "falling"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current premium", f"${lp:,.0f}")
        m2.metric("3-month avg",     f"${ma3:,.0f}")
        m3.metric("12-month avg",    f"${ma12:,.0f}")
        m4.metric("Trend",           f"{'Up' if trend == 'rising' else 'Down'} {trend}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["month"], y=df["premium"], name="Premium",
            line=dict(color=c["color"], width=1.8),
            hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=df["month"], y=df["ma3"], name="3M avg",
            line=dict(color="#fbbf24", width=1.4, dash="dot")))
        fig.add_trace(go.Scatter(x=df["month"], y=df["ma12"], name="12M avg",
            line=dict(color="#f472b6", width=1.4, dash="dash")))
        fig.update_layout(**PLOT, height=340, hovermode="x unified",
                          yaxis=dict(tickprefix="$", tickformat=",", gridcolor="#13131f", zerolinecolor="#1e1e30"),
                          xaxis=dict(gridcolor="#13131f"),
                          legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

        pct_color = "#ef4444" if pct > 75 else ("#22c55e" if pct < 35 else "#f59e0b")
        st.markdown(f'<div class="section-label">Historical percentile</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
            <span style="font-size:1.5rem;font-weight:700;color:{pct_color};">{pct:.0f}%</span>
            <span style="font-size:0.82rem;color:#475569;">of historical premiums were below today's price</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(pct / 100)

        if pct > 75:
            st.markdown(f'<div class="warn">Premiums are historically elevated and {trend}. Consider waiting if your timeline allows.</div>', unsafe_allow_html=True)
        elif pct < 35:
            st.markdown(f'<div class="good">Premiums are historically low and {trend}. This may be a good window to bid.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="insight">Premiums are in a neutral range. Let your personal needs drive the decision.</div>', unsafe_allow_html=True)

# PAGE 6 - Affordability Calculator
elif page == "Affordability Calculator":
    st.markdown('<div class="page-title">Affordability Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Estimate monthly repayments and assess financial readiness</div>', unsafe_allow_html=True)

    latest = get_latest()
    lmap   = {d["category"]: d["premium"] for d in latest}

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="section-label">Vehicle details</div>', unsafe_allow_html=True)
        seg = st.selectbox("Segment", [
            "Mass market - Cat A (Toyota, Honda, Mazda)",
            "Premium - Cat B/E (BMW 3, Mercedes C, Audi A4)",
            "Luxury - Cat B/E (BMW 5+, Porsche, Lexus)"
        ])
        coe_cat = "Category A" if "Mass" in seg else ("Category B" if "Premium" in seg else "Category E")
        default_price = 110000 if "Mass" in seg else (200000 if "Premium" in seg else 350000)
        car_price  = st.number_input("Car price excl. COE (SGD)", value=default_price, step=5000, format="%d")
        coe_val    = int(lmap.get(coe_cat, 100000))
        coe_input  = st.number_input(f"COE premium - {coe_cat} (SGD)", value=coe_val, step=1000, format="%d")
        loan_pct   = st.slider("Loan percentage", 0, 70, 60, help="MAS cap: 70% for OMV under 20k, 60% above")
        tenure     = st.slider("Tenure (years)", 1, 7, 5)
        rate       = st.slider("Interest rate % per year", 1.5, 5.0, 2.78, 0.01)
        st.markdown('<div class="section-label">Your finances</div>', unsafe_allow_html=True)
        income     = st.number_input("Monthly gross income (SGD)", value=6000, step=500, format="%d")
        other_debt = st.number_input("Other monthly loan repayments (SGD)", value=0, step=100, format="%d")

    with c2:
        total    = car_price + coe_input
        loan     = total * loan_pct / 100
        dp       = total - loan
        n        = tenure * 12
        r        = (rate / 100) / 12
        monthly  = loan * r * (1 + r) ** n / ((1 + r) ** n - 1) if r > 0 else loan / n
        interest = monthly * n - loan
        dsr      = (monthly + other_debt) / income * 100
        inc_need = (monthly + other_debt) / 0.30

        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Total OTR cost", f"${total:,.0f}")
        m2.metric("Downpayment needed", f"${dp:,.0f}")
        m3, m4 = st.columns(2)
        m3.metric("Monthly repayment", f"${monthly:,.0f}")
        m4.metric("Total interest paid", f"${interest:,.0f}")

        st.markdown('<div class="section-label">Debt service ratio</div>', unsafe_allow_html=True)
        dsr_color = "#22c55e" if dsr < 30 else ("#f59e0b" if dsr < 40 else "#ef4444")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
            <span style="font-size:1.6rem;font-weight:700;color:{dsr_color};">{dsr:.1f}%</span>
            <span style="font-size:0.8rem;color:#475569;">of monthly income committed to loans</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(dsr / 60, 1.0))

        if dsr < 30:
            st.markdown('<div class="good">DSR is healthy. Monthly commitments are within the 30% guideline.</div>', unsafe_allow_html=True)
        elif dsr < 40:
            st.markdown('<div class="warn">DSR is stretched. Consider a larger downpayment or lower-priced vehicle.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn">DSR exceeds 40%. This purchase may cause financial stress.</div>', unsafe_allow_html=True)

        st.metric("Recommended minimum income", f"${inc_need:,.0f} per month")

        st.markdown('<div class="section-label">Loan amortisation</div>', unsafe_allow_html=True)
        rows = []
        bal = loan
        for yr in range(1, tenure + 1):
            yr_int = 0
            yr_prin = 0
            for _ in range(12):
                i = bal * r
                p = monthly - i
                yr_int += i
                yr_prin += p
                bal -= p
            rows.append({"Year": yr, "Principal": round(yr_prin), "Interest": round(yr_int), "Balance": round(max(bal, 0))})
        df_am = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_am["Year"], y=df_am["Principal"], name="Principal", marker_color="#818cf8", marker_line_width=0))
        fig.add_trace(go.Bar(x=df_am["Year"], y=df_am["Interest"],  name="Interest",  marker_color="#f472b6", marker_line_width=0))
        fig.update_layout(**PLOT, barmode="stack", height=220,
                          yaxis=dict(tickprefix="$", tickformat=",", gridcolor="#13131f"),
                          xaxis=dict(title="Year", gridcolor="#13131f"),
                          legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

# PAGE 7 - Total Cost of Ownership
elif page == "Total Cost of Ownership":
    st.markdown('<div class="page-title">Total Cost of Ownership</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Full cost breakdown beyond the sticker price</div>', unsafe_allow_html=True)

    latest = get_latest()
    lmap   = {d["category"]: d["premium"] for d in latest}

    c1, c2 = st.columns(2, gap="large")
    with c1:
        car_price = st.number_input("Car price excl. COE (SGD)", value=120000, step=5000, format="%d")
        coe_cat   = st.selectbox("COE category", list(CATS.keys()))
        coe_val   = int(lmap.get(coe_cat, 100000))
        st.caption(f"Current {coe_cat} COE: ${coe_val:,.0f}")
        petrol    = st.number_input("Monthly petrol or charging (SGD)", value=200, step=10)
        insurance = st.number_input("Annual insurance (SGD)", value=1800, step=100)
        parking   = st.number_input("Monthly parking (SGD)", value=150, step=10)
        servicing = st.number_input("Annual servicing (SGD)", value=800, step=100)
        road_tax  = st.number_input("Annual road tax (SGD)", value=742, step=10)
        erp       = st.number_input("Monthly ERP (SGD)", value=80, step=10)
        yrs       = st.slider("Ownership period (years)", 1, 10, 10)

    with c2:
        t_car  = car_price + coe_val
        t_fuel = petrol * 12 * yrs
        t_ins  = insurance * yrs
        t_park = parking * 12 * yrs
        t_svc  = servicing * yrs
        t_rt   = road_tax * yrs
        t_erp  = erp * 12 * yrs
        total  = t_car + t_fuel + t_ins + t_park + t_svc + t_rt + t_erp
        mth    = total / (yrs * 12)

        st.markdown(f"""
        <div class="kpi-wrap" style="margin-bottom:1rem;">
            <div style="position:absolute;top:0;left:0;right:0;height:2px;background:#818cf8;border-radius:14px 14px 0 0;"></div>
            <div class="kpi-cat" style="color:#818cf8;">{yrs}-year total cost of ownership</div>
            <div class="kpi-val" style="color:#818cf8;">${total:,.0f}</div>
            <div class="kpi-meta">approximately <b style="color:#94a3b8">${mth:,.0f}</b> per month all-in</div>
        </div>
        """, unsafe_allow_html=True)

        items  = {"Car + COE": t_car, "Petrol/Charging": t_fuel, "Insurance": t_ins,
                  "Parking": t_park, "Servicing": t_svc, "Road Tax": t_rt, "ERP": t_erp}
        colors = ["#818cf8", "#f472b6", "#34d399", "#fb923c", "#a78bfa", "#fbbf24", "#38bdf8"]
        fig = go.Figure(go.Pie(
            labels=list(items.keys()), values=list(items.values()),
            hole=0.55, marker_colors=colors, marker_line_width=0,
        ))
        fig.update_layout(**PLOT, height=280, showlegend=True,
                          legend=dict(orientation="v", x=1.02, y=0.5, bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

        for name, val, col in zip(items.keys(), items.values(), colors):
            pct = val / total * 100
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:5px;">
                <div style="width:8px;height:8px;border-radius:50%;background:{col};flex-shrink:0;"></div>
                <div style="font-size:0.8rem;color:#64748b;flex:1;">{name}</div>
                <div style="font-size:0.8rem;font-weight:500;color:#94a3b8;">${val:,.0f}</div>
                <div style="font-size:0.75rem;color:#475569;min-width:36px;text-align:right;">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

# PAGE 8 - Renew vs Scrap
elif page == "Renew vs Scrap":
    st.markdown('<div class="page-title">Renew vs Scrap</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Should you renew your COE or scrap and buy new?</div>', unsafe_allow_html=True)

    latest = get_latest()
    lmap   = {d["category"]: d["premium"] for d in latest}

    c1, c2 = st.columns(2, gap="large")
    with c1:
        veh_cat = st.selectbox("Vehicle category", ["Category A", "Category B", "Category D"])
        omv     = st.number_input("Original Market Value - OMV (SGD)", value=20000, step=1000)
        age     = st.slider("Current vehicle age (years)", 5, 10, 10)
        renew_y = st.radio("Renewal period", [5, 10], horizontal=True)

    with c2:
        cur_coe = int(lmap.get(veh_cat, 80000))
        renewal = cur_coe * (0.5 if renew_y == 5 else 1.0)
        parf    = omv * max(0, (10 - age) / 10)
        scrap   = parf + 2000
        new_tot = 100000 + cur_coe
        saving  = new_tot - renewal

        m1, m2 = st.columns(2)
        m1.metric(f"COE renewal ({renew_y}yr)", f"${renewal:,.0f}")
        m2.metric("Scrap value (PARF + metal)", f"${scrap:,.0f}")
        m3, m4 = st.columns(2)
        m3.metric("Est. new car total cost", f"${new_tot:,.0f}")
        m4.metric("Savings vs buying new", f"${saving:,.0f}")

        fig = go.Figure(go.Bar(
            x=["COE Renewal", "Scrap Value", "New Car Cost"],
            y=[renewal, scrap, new_tot],
            marker_color=["#818cf8", "#34d399", "#f472b6"],
            marker_line_width=0,
            text=[f"${v:,.0f}" for v in [renewal, scrap, new_tot]],
            textposition="outside", textfont=dict(size=12),
        ))
        fig.update_layout(**PLOT, height=280,
                          yaxis=dict(tickprefix="$", tickformat=",", gridcolor="#13131f", showticklabels=False),
                          xaxis=dict(gridcolor="#13131f"))
        st.plotly_chart(fig, use_container_width=True)

        if renewal < 60000 and renewal < new_tot * 0.45:
            st.markdown('<div class="good">Renewal is financially sensible at current COE prices.</div>', unsafe_allow_html=True)
        elif renewal > 90000:
            st.markdown('<div class="warn">Renewal cost is high. Scrapping or waiting for lower COE may be worth considering.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight">Close call. Factor in your vehicle condition and which direction COE premiums are trending.</div>', unsafe_allow_html=True)