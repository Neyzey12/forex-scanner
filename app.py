import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="FX Chart Scanner",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# THEME — dark purple neon
# ============================================================

CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at 20% 0%, #2a0a4a 0%, #15042a 40%, #0a0118 100%);
        color: #e9e4ff;
    }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 720px;}

    .app-title {
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        background: linear-gradient(90deg, #c084fc, #7c3aed, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .app-sub {
        text-align: center;
        color: #a78bfa;
        font-size: 0.85rem;
        margin-bottom: 1.2rem;
    }
    .badge-running {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(34,197,94,0.15);
        color: #22c55e;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid rgba(34,197,94,0.4);
    }

    .signal-card {
        background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(34,211,238,0.06));
        border: 1px solid rgba(168,85,247,0.35);
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 0 24px rgba(124,58,237,0.25);
    }
    .signal-card.buy {
        border-color: rgba(34,197,94,0.6);
        box-shadow: 0 0 26px rgba(34,197,94,0.35);
    }
    .signal-card.sell {
        border-color: rgba(239,68,68,0.6);
        box-shadow: 0 0 26px rgba(239,68,68,0.35);
    }

    .sig-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .sig-pair { font-size: 1.15rem; font-weight: 800; color: #ffffff; }
    .sig-tf {
        font-size: 0.7rem; color: #a78bfa; margin-left: 6px;
        border: 1px solid rgba(167,139,250,0.4);
        padding: 1px 6px; border-radius: 6px;
    }

    .stamp {
        font-weight: 900;
        font-size: 1.05rem;
        letter-spacing: 2px;
        padding: 6px 16px;
        border-radius: 10px;
    }
    .stamp-buy  { background: rgba(34,197,94,0.2);  color: #22c55e; border: 1.5px solid #22c55e;
                  box-shadow: 0 0 14px rgba(34,197,94,0.55); }
    .stamp-sell { background: rgba(239,68,68,0.2);  color: #ef4444; border: 1.5px solid #ef4444;
                  box-shadow: 0 0 14px rgba(239,68,68,0.55); }
    .stamp-wait { background: rgba(148,163,184,0.15); color: #94a3b8; border: 1.5px solid rgba(148,163,184,0.5); }

    .sig-meta {
        font-size: 0.8rem;
        color: #cbd5e1;
        line-height: 1.6;
    }
    .sig-meta b { color: #c084fc; }
    .sig-setup {
        font-size: 0.75rem; color: #94a3b8;
        margin-top: 6px; font-style: italic;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #7c3aed, #a855f7);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 10px 0;
        box-shadow: 0 0 18px rgba(124,58,237,0.5);
    }
    .stButton > button:hover { background: linear-gradient(90deg, #a855f7, #22d3ee); }

    .streamlit-expanderHeader { color: #c084fc !important; }
    label { color: #c084fc !important; font-weight: 600; }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7c3aed, #22d3ee);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# Config — forex + metals + indices + energy
# ============================================================

PAIRS = {
    # --- Forex majors ---
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X",
    # --- Forex crosses ---
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "EUR/GBP": "EURGBP=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/CHF": "EURCHF=X",
    # --- Metals ---
    "XAU/USD": "GC=F",
    "XAG/USD": "SI=F",
    # --- Indices ---
    "NAS100":  "^NDX",
    "US30":    "^DJI",
    "SPX500":  "^GSPC",
    # --- Energy ---
    "USOIL":   "CL=F",
}

# Instruments that show fewer decimals (indices, oil)
FEW_DECIMALS = {"NAS100", "US30", "SPX500", "USOIL"}


def fmt_price(label: str, value: float) -> str:
    if value is None:
        return "—"
    if label in FEW_DECIMALS:
        return f"{value:,.2f}"
    return f"{value:.5f}"


TF_CONFIG = {
    "1h": {"interval": "60m", "period": "60d", "resample": None},
    "4h": {"interval": "60m", "period": "60d", "resample": "4h"},
    "1d": {"interval": "1d", "period": "2y", "resample": None},
}

P = dict(ema_fast=20, ema_slow=50, ema_trend=200,
         rsi_len=14, atr_len=14, don_len=20,
         buy_th=4, sell_th=-4)

# ============================================================
# Data
# ============================================================

def fetch_ohlc(symbol, tf):
    cfg = TF_CONFIG[tf]
    raw = yf.Ticker(symbol).history(period=cfg["period"],
                                    interval=cfg["interval"],
                                    auto_adjust=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw[["Open", "High", "Low", "Close"]].dropna()
    if cfg["resample"]:
        df = (df.resample(cfg["resample"])
                .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
                .dropna())
    return df


@st.cache_data(ttl=60, show_spinner=False)
def get_data(symbol, tf):
    return fetch_ohlc(symbol, tf)

# ============================================================
# Indicators
# ============================================================

def indicators(df, p):
    d = df.copy()
    c = d["Close"]
    d["EMA_f"] = c.ewm(span=p["ema_fast"], adjust=False).mean()
    d["EMA_s"] = c.ewm(span=p["ema_slow"], adjust=False).mean()
    d["EMA_t"] = c.ewm(span=p["ema_trend"], adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/p["rsi_len"], adjust=False).mean()
    al = loss.ewm(alpha=1/p["rsi_len"], adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    d["RSI"] = (100 - 100/(1+rs)).where(al != 0, 100.0)

    pc = c.shift()
    tr = pd.concat([d["High"]-d["Low"],
                    (d["High"]-pc).abs(),
                    (d["Low"]-pc).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1/p["atr_len"], adjust=False).mean()

    d["DC_H"] = d["High"].shift(1).rolling(p["don_len"]).max()
    d["DC_L"] = d["Low"].shift(1).rolling(p["don_len"]).min()

    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    d["MACD_H"] = macd - macd.ewm(span=9, adjust=False).mean()
    return d

# ============================================================
# Scoring — outputs BUY / SELL / WAIT
# ============================================================

def analyze(df, p):
    if df is None or len(df) < p["ema_trend"] + 10:
        return None
    d = indicators(df, p)
    last = d.iloc[-1]

    score = 0
    reasons = []

    if last.EMA_f > last.EMA_s > last.EMA_t:
        score += 2; reasons.append("EMA stack bullish")
    elif last.EMA_f < last.EMA_s < last.EMA_t:
        score -= 2; reasons.append("EMA stack bearish")

    score += 1 if last.Close > last.EMA_t else -1

    if last.RSI >= 55:
        score += 1; reasons.append(f"RSI {last.RSI:.0f} bullish")
    elif last.RSI <= 45:
        score -= 1; reasons.append(f"RSI {last.RSI:.0f} bearish")

    score += 1 if last.MACD_H > 0 else -1

    if last.Close > last.DC_H:
        score += 2; reasons.append(f"{p['don_len']}-bar breakout up")
    elif last.Close < last.DC_L:
        score -= 2; reasons.append(f"{p['don_len']}-bar breakout down")

    score += 1 if last.Close > d["Close"].iloc[-6] else -1

    if score >= p["buy_th"]:
        signal = "BUY"
    elif score <= p["sell_th"]:
        signal = "SELL"
    else:
        signal = "WAIT"

    atr_pct = (last.ATR / last.Close * 100) if last.Close else np.nan
    atr_val = float(last.ATR) if not pd.isna(last.ATR) else 0.0
    price = float(last.Close)

    if signal == "BUY":
        sl, tp = price - 1.5 * atr_val, price + 3.0 * atr_val
    elif signal == "SELL":
        sl, tp = price + 1.5 * atr_val, price - 3.0 * atr_val
    else:
        sl = tp = None

    return dict(Signal=signal, Score=int(score),
                Price=price, RSI=round(float(last.RSI), 1),
                ATR_pct=round(float(atr_pct), 2),
                ATR=atr_val, SL=sl, TP=tp,
                Setup=", ".join(reasons) if reasons else "—")

# ============================================================
# UI — header
# ============================================================

st.markdown('<div class="app-title">🔮 FX Chart Scanner</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Buy · Sell · Forex · Metals · Indices</div>', unsafe_allow_html=True)

with st.expander("⚙️ Scanner Settings", expanded=False):
    pair_sel = st.multiselect(
        "Pairs",
        options=list(PAIRS.keys()),
        default=["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "NAS100", "US30"],
    )
    tf_sel = st.multiselect(
        "Timeframes",
        options=list(TF_CONFIG.keys()),
        default=["1h", "4h", "1d"],
    )
    only_actionable = st.checkbox("Show only BUY / SELL", value=True)

col_a, col_b = st.columns(2)
with col_a: refresh = st.button("🔄 Refresh")
with col_b: run_scan = st.button("🚀 Scan")

st.markdown(
    '<div style="text-align:center;margin:6px 0 14px 0;">'
    '<span class="badge-running">● SCANNER READY</span></div>',
    unsafe_allow_html=True,
)

if refresh:
    get_data.clear()
    st.rerun()

# ============================================================
# Run scan
# ============================================================

if not pair_sel or not tf_sel:
    st.info("Pick at least one pair and one timeframe above.")
    st.stop()

if run_scan or "scanned" not in st.session_state:
    st.session_state.scanned = True
    rows, errors = [], []
    prog = st.progress(0.0, text="Scanning markets…")
    total = len(pair_sel) * len(tf_sel)
    done = 0
    for label in pair_sel:
        sym = PAIRS[label]
        for tf in tf_sel:
            done += 1
            prog.progress(done / total, text=f"Analyzing {label} · {tf}")
            try:
                res = analyze(get_data(sym, tf), P)
                if res:
                    rows.append({"Pair": label, "TF": tf, **res})
            except Exception as e:
                errors.append(f"{label} {tf}: {e}")
    prog.empty()
    st.session_state.results = pd.DataFrame(rows)
    st.session_state.errors = errors

results = st.session_state.get("results", pd.DataFrame())

if results.empty:
    st.warning("No data returned. Try Refresh, or fewer pairs.")
    st.stop()

# ============================================================
# Summary
# ============================================================

n_buy = int((results["Signal"] == "BUY").sum())
n_sell = int((results["Signal"] == "SELL").sum())
n_wait = int((results["Signal"] == "WAIT").sum())

m1, m2, m3 = st.columns(3)
m1.markdown(f"<div style='text-align:center'><div style='color:#22c55e;font-size:1.5rem;font-weight:900'>{n_buy}</div><div style='color:#94a3b8;font-size:0.75rem;letter-spacing:1px'>BUY</div></div>", unsafe_allow_html=True)
m2.markdown(f"<div style='text-align:center'><div style='color:#ef4444;font-size:1.5rem;font-weight:900'>{n_sell}</div><div style='color:#94a3b8;font-size:0.75rem;letter-spacing:1px'>SELL</div></div>", unsafe_allow_html=True)
m3.markdown(f"<div style='text-align:center'><div style='color:#94a3b8;font-size:1.5rem;font-weight:900'>{n_wait}</div><div style='color:#94a3b8;font-size:0.75rem;letter-spacing:1px'>WAIT</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# Signal cards
# ============================================================

view = results.copy()
if only_actionable:
    view = view[view["Signal"].isin(["BUY", "SELL"])]
view = view.sort_values("Score", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

if view.empty:
    st.info("No BUY or SELL setups right now. Untick 'Show only BUY / SELL' to see WAIT signals.")
else:
    st.markdown("### 📡 Live Signals")
    for _, r in view.iterrows():
        cls = "buy" if r.Signal == "BUY" else "sell" if r.Signal == "SELL" else ""
        stamp = {"BUY": "stamp-buy", "SELL": "stamp-sell"}.get(r.Signal, "stamp-wait")

        entry_txt = fmt_price(r.Pair, r.Price)
        if r.Signal in ("BUY", "SELL") and r.SL is not None:
            sl_txt = fmt_price(r.Pair, r.SL)
            tp_txt = fmt_price(r.Pair, r.TP)
            levels_html = (
                f"<div class='sig-meta'>"
                f"Entry <b>{entry_txt}</b> &nbsp;·&nbsp; "
                f"SL <b>{sl_txt}</b> &nbsp;·&nbsp; "
                f"TP <b>{tp_txt}</b>"
                f"</div>"
            )
        else:
            levels_html = f"<div class='sig-meta'>Price <b>{entry_txt}</b></div>"

        st.markdown(f"""
        <div class="signal-card {cls}">
          <div class="sig-row">
            <div><span class="sig-pair">{r.Pair}</span>
                 <span class="sig-tf">{r.TF}</span></div>
            <div><span class="stamp {stamp}">{r.Signal}</span></div>
          </div>
          {levels_html}
          <div class="sig-meta">
            Score <b>{r.Score:+d}</b> &nbsp;·&nbsp;
            RSI <b>{r.RSI:.1f}</b> &nbsp;·&nbsp;
            ATR% <b>{r.ATR_pct:.2f}</b>
          </div>
          <div class="sig-setup">{r.Setup}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# Chart
# ============================================================

st.markdown("### 📈 Chart")

if not view.empty:
    chart_options = [f"{r.Pair} · {r.TF}" for _, r in view.iterrows()]
    choice = st.selectbox("Select pair", chart_options, label_visibility="collapsed")
    sel_pair, sel_tf = choice.split(" · ")

    d = indicators(get_data(PAIRS[sel_pair], sel_tf), P).tail(300)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.04)

    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"],
        low=d["Low"], close=d["Close"], name="Price",
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
    ), row=1, col=1)

    for col, color, name in [("EMA_f", "#60a5fa", "EMA20"),
                             ("EMA_s", "#f59e0b", "EMA50"),
                             ("EMA_t", "#a855f7", "EMA200")]:
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=name,
                                 line=dict(width=1.4, color=color)),
                      row=1, col=1)

    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI",
                             line=dict(width=1.5, color="#22d3ee")),
                  row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#ef4444", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#22c55e", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#64748b", row=2, col=1)

    fig.update_layout(
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,4,42,0.6)",
        font=dict(color="#e9e4ff"),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.08)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.08)")

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ============================================================
# Footer
# ============================================================

if st.session_state.get("errors"):
    with st.expander(f"⚠️ {len(st.session_state.errors)} fetch error(s)"):
        for e in st.session_state.errors:
            st.text(e)

st.markdown(
    '<div style="text-align:center;color:#64748b;font-size:0.7rem;margin-top:1.5rem;">'
    'Data: Yahoo Finance (delayed) · Educational only · Not financial advice'
    '</div>',
    unsafe_allow_html=True,
)