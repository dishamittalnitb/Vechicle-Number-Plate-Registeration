import streamlit as st
import pandas as pd
import time
import os
from PIL import Image
from datetime import datetime


st.set_page_config(
    page_title="AVRS Command Center",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');

:root {
    --bg-primary:  #080d14;
    --bg-card:     #0d1624;
    --accent-cyan: #00d4ff;
    --accent-green:#00ff88;
    --accent-amber:#ffb300;
    --accent-red:  #ff3d5a;
    --text-primary:#e8f4fd;
    --text-muted:  #5a7fa0;
    --border:      rgba(0,212,255,0.15);
    --border-hi:   rgba(0,212,255,0.45);
    --glow-cyan:   0 0 20px rgba(0,212,255,0.3);
    --glow-green:  0 0 20px rgba(0,255,136,0.3);
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    font-family: 'Exo 2', sans-serif;
    color: var(--text-primary);
}
[data-testid="stMain"] { background: transparent !important; }
[data-testid="block-container"] { padding: 1.5rem 2rem 2rem; max-width:100% !important; }

.avrs-header {
    display:flex; align-items:center; gap:1.2rem;
    padding:1rem 1.6rem;
    background:linear-gradient(135deg,rgba(0,212,255,0.08) 0%,rgba(0,0,0,0) 60%);
    border:1px solid var(--border-hi);
    border-radius:12px; margin-bottom:1.4rem;
    position:relative; overflow:hidden;
}
.avrs-header::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,transparent,var(--accent-cyan),var(--accent-green),transparent);
    animation:scanline 3s linear infinite;
}
@keyframes scanline { 0%,100%{opacity:0.4} 50%{opacity:1} }
.avrs-logo { font-size:2.4rem; line-height:1; }
.avrs-title-block { flex:1; }
.avrs-title {
    font-family:'Rajdhani',sans-serif; font-size:1.9rem; font-weight:700;
    color:var(--accent-cyan); letter-spacing:0.12em; text-transform:uppercase;
    margin:0; text-shadow:var(--glow-cyan);
}
.avrs-subtitle {
    font-family:'Share Tech Mono',monospace; font-size:0.72rem;
    color:var(--text-muted); letter-spacing:0.18em; text-transform:uppercase; margin:0;
}
.avrs-status-pill {
    display:flex; align-items:center; gap:0.5rem;
    padding:0.4rem 1rem; border-radius:30px;
    background:rgba(0,255,136,0.08); border:1px solid rgba(0,255,136,0.3);
    font-family:'Share Tech Mono',monospace; font-size:0.75rem;
    color:var(--accent-green); letter-spacing:0.1em;
}
.pulse-dot {
    width:8px; height:8px; border-radius:50%;
    background:var(--accent-green); box-shadow:var(--glow-green);
    animation:pulse 1.6s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.75)} }
.avrs-clock {
    font-family:'Share Tech Mono',monospace; font-size:0.85rem;
    color:var(--text-muted); letter-spacing:0.08em;
}

.metric-card {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:10px; padding:1.1rem 1.3rem;
    position:relative; overflow:hidden;
}
.metric-card::after {
    content:''; position:absolute; top:0; left:0; right:0;
    height:2px; border-radius:10px 10px 0 0;
}
.metric-card.cyan::after  { background:var(--accent-cyan);  box-shadow:var(--glow-cyan); }
.metric-card.green::after { background:var(--accent-green); box-shadow:var(--glow-green); }
.metric-card.amber::after { background:var(--accent-amber); }
.metric-label {
    font-family:'Share Tech Mono',monospace; font-size:0.65rem;
    letter-spacing:0.18em; text-transform:uppercase;
    color:var(--text-muted); margin-bottom:0.4rem;
}
.metric-value {
    font-family:'Rajdhani',sans-serif; font-size:2.1rem; font-weight:700;
    line-height:1; margin-bottom:0.2rem;
}
.metric-value.cyan  { color:var(--accent-cyan);  text-shadow:var(--glow-cyan); }
.metric-value.green { color:var(--accent-green); text-shadow:var(--glow-green); }
.metric-value.amber { color:var(--accent-amber); }
.metric-sub { font-size:0.72rem; color:var(--text-muted); }

.det-card {
    background:rgba(0,212,255,0.04);
    border:1px solid var(--border);
    border-left:3px solid var(--accent-cyan);
    border-radius:8px; padding:0.75rem 1rem; margin-bottom:0.6rem;
}
.det-plate {
    font-family:'Share Tech Mono',monospace; font-size:1.05rem;
    color:var(--accent-cyan); letter-spacing:0.1em; font-weight:700;
}
.det-time {
    font-family:'Exo 2',sans-serif; font-size:0.72rem;
    color:var(--text-muted); margin-top:0.2rem;
}
.badge {
    display:inline-block; padding:0.1rem 0.5rem; border-radius:4px;
    font-size:0.65rem; font-weight:600; letter-spacing:0.1em;
    text-transform:uppercase; margin-left:0.5rem;
}
.badge-in  { background:rgba(0,255,136,0.12); color:var(--accent-green); border:1px solid rgba(0,255,136,0.25); }
.badge-out { background:rgba(255,61,90,0.12); color:var(--accent-red);   border:1px solid rgba(255,61,90,0.25); }
.badge-unk { background:rgba(90,127,160,0.15);color:var(--text-muted);   border:1px solid var(--border); }

.sec-div {
    display:flex; align-items:center; gap:1rem; margin:1.2rem 0 1rem;
}
.sec-line { flex:1; height:1px; background:linear-gradient(90deg,var(--border-hi),transparent); }
.sec-label {
    font-family:'Rajdhani',sans-serif; font-size:1rem; font-weight:600;
    color:var(--accent-cyan); letter-spacing:0.12em; text-transform:uppercase;
}
.rec-label {
    font-family:'Share Tech Mono',monospace; font-size:0.65rem;
    letter-spacing:0.14em; color:var(--accent-green); text-align:center;
    padding:0.35rem; background:rgba(0,255,136,0.06);
    border:1px solid rgba(0,255,136,0.15); border-top:none; border-radius:0 0 6px 6px;
}
.panel-title-bar {
    font-family:'Rajdhani',sans-serif; font-size:0.9rem; font-weight:600;
    color:var(--accent-cyan); letter-spacing:0.1em; text-transform:uppercase;
    padding:0.6rem 0; border-bottom:1px solid var(--border); margin-bottom:0.8rem;
}
.no-feed {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    height:200px; gap:0.8rem; color:var(--text-muted);
    font-family:'Share Tech Mono',monospace; font-size:0.75rem; letter-spacing:0.1em;
    border:2px dashed rgba(90,127,160,0.2); border-radius:8px;
}
.waiting {
    text-align:center; padding:2rem; color:var(--text-muted);
    font-family:'Share Tech Mono',monospace; font-size:0.8rem; letter-spacing:0.1em;
    border:1px dashed rgba(90,127,160,0.25); border-radius:10px;
}

[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:10px !important; }
[data-testid="stDataFrame"] th {
    background:rgba(0,212,255,0.07) !important; color:var(--accent-cyan) !important;
    font-family:'Rajdhani',sans-serif !important; font-size:0.8rem !important;
    letter-spacing:0.1em !important; text-transform:uppercase !important;
}
[data-testid="stDataFrame"] td { color:var(--text-primary) !important; }

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
[data-testid="stHeader"] { background:transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ────────────────────────────────────────────────────────
def badge_html(direction: str) -> str:
    d = str(direction).strip().upper()
    if "IN" in d:
        return '<span class="badge badge-in">↓ IN</span>'
    elif "OUT" in d:
        return '<span class="badge badge-out">↑ OUT</span>'
    return f'<span class="badge badge-unk">{direction}</span>'

def metric_card(label, value, sub="", color="cyan"):
    return f"""<div class="metric-card {color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>"""


# ── STATIC HEADER ──────────────────────────────────────────────────
now_str = datetime.now().strftime("%d %b %Y")
st.markdown(f"""
<div class="avrs-header">
    <div class="avrs-logo">🚗</div>
    <div class="avrs-title-block">
        <div class="avrs-title">AVRS Command Center</div>
        <div class="avrs-subtitle">Automated Vehicle Registration System — Live Monitoring</div>
    </div>
    <div style="display:flex;align-items:center;gap:1rem;">
        <div class="avrs-status-pill"><div class="pulse-dot"></div>ONLINE</div>
        <div class="avrs-clock">{now_str}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── LAYOUT PLACEHOLDERS ─────────────────────────────────────────────
metrics_ph = st.empty()
st.write("")

vid_col, log_col = st.columns([4, 6], gap="medium")
with vid_col:
    feed_ph = st.empty()
with log_col:
    cards_col, img_col = st.columns([3, 2], gap="small")
    with cards_col:
        logs_ph = st.empty()
    with img_col:
        imgs_ph = st.empty()

st.markdown("""
<div class="sec-div">
    <div class="sec-line"></div>
    <div class="sec-label">🗄 Vehicle Logs Database</div>
    <div class="sec-line" style="background:linear-gradient(90deg,transparent,rgba(0,212,255,0.45))"></div>
</div>
""", unsafe_allow_html=True)

table_ph = st.empty()


# ── RENDER FUNCTIONS ────────────────────────────────────────────────

def render_metrics(df, now_time):
    total     = len(df)
    last_time = df["Time"].iloc[-1] if not df.empty else "—"
    last_gate = df["Gate"].iloc[-1] if "Gate" in df.columns else "Main Gate"
    n_in  = df["Direction"].str.upper().str.contains("IN",  na=False).sum() if "Direction" in df.columns else "—"
    n_out = df["Direction"].str.upper().str.contains("OUT", na=False).sum() if "Direction" in df.columns else "—"
    with metrics_ph.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Total Vehicles", str(total),           f"As of {now_time}", "cyan"),  unsafe_allow_html=True)
        c2.markdown(metric_card("Last Activity",  str(last_time),       str(last_gate),      "amber"), unsafe_allow_html=True)
        c3.markdown(metric_card("System Status",  "ACTIVE",             "All sensors nominal","green"),unsafe_allow_html=True)
        c4.markdown(metric_card("In / Out",       f"{n_in} / {n_out}",  "Direction split",   "cyan"),  unsafe_allow_html=True)


def render_metrics_empty(now_time):
    with metrics_ph.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Total Vehicles", "0",     "Waiting for data", "cyan"),  unsafe_allow_html=True)
        c2.markdown(metric_card("Last Activity",  "—",     "No events yet",    "amber"), unsafe_allow_html=True)
        c3.markdown(metric_card("System Status",  "ACTIVE","Sensors ready",    "green"), unsafe_allow_html=True)
        c4.markdown(metric_card("In / Out",       "— / —", "",                 "cyan"),  unsafe_allow_html=True)


def render_feed(now_time):
    # KEY RULE: st.image() is a plain widget call inside with container().
    # It is NEVER placed between HTML open/close tags.
    with feed_ph.container():
        st.markdown('<div class="panel-title-bar">📷 &nbsp;Main Gate — Live Feed</div>', unsafe_allow_html=True)
        if os.path.exists("latest_frame.jpg"):
            try:
                frame = Image.open("latest_frame.jpg")
                st.image(frame, use_container_width=True)
                st.markdown(
                    f'<div class="rec-label">● REC &nbsp;|&nbsp; {now_time} &nbsp;|&nbsp; GATE-01</div>',
                    unsafe_allow_html=True,
                )
                return
            except Exception:
                pass
        st.markdown(
            '<div class="no-feed"><span style="font-size:2rem;opacity:0.3">📷</span>AWAITING CAMERA SIGNAL...</div>',
            unsafe_allow_html=True,
        )


def render_detections(df):
    recent_df = df.tail(5).iloc[::-1]

    # ── Cards column (pure HTML, no widgets) ──
    cards = ""
    for _, row in recent_df.iterrows():
        plate = row.get("Plate Number", "UNKNOWN")
        t     = row.get("Time", "")
        direc = row.get("Direction", "")
        cards += f"""<div class="det-card">
            <div class="det-plate">{plate}</div>
            <div class="det-time">🕐 {t} {badge_html(direc)}</div>
        </div>"""

    with logs_ph.container():
        st.markdown('<div class="panel-title-bar">🔍 &nbsp;Recent Detections</div>', unsafe_allow_html=True)
        st.markdown(cards, unsafe_allow_html=True)

    # ── Images column — native st.image calls in their own placeholder ──
    with imgs_ph.container():
        st.markdown('<div class="panel-title-bar">🖼 &nbsp;Plates</div>', unsafe_allow_html=True)
        any_img = False
        for _, row in recent_df.iterrows():
            img_path = str(row.get("Image Path", ""))
            if img_path and os.path.exists(img_path):
                try:
                    st.image(Image.open(img_path), caption=str(row.get("Plate Number", "")), use_container_width=True)
                    any_img = True
                except Exception:
                    pass
        if not any_img:
            st.markdown('<div style="color:var(--text-muted);font-family:\'Share Tech Mono\',monospace;font-size:0.72rem;padding:0.5rem 0;">No images yet</div>', unsafe_allow_html=True)


def render_detections_empty():
    with logs_ph.container():
        st.markdown('<div class="panel-title-bar">🔍 &nbsp;Recent Detections</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="waiting"><div style="font-size:2rem;opacity:0.3">⏳</div>AWAITING FIRST DETECTION...</div>',
            unsafe_allow_html=True,
        )
    with imgs_ph.container():
        st.markdown('<div class="panel-title-bar">🖼 &nbsp;Plates</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:var(--text-muted);font-family:\'Share Tech Mono\',monospace;font-size:0.72rem;padding:0.5rem 0;">No images yet</div>', unsafe_allow_html=True)


# ── REAL-TIME LOOP ──────────────────────────────────────────────────
while True:
    now_time = datetime.now().strftime("%H:%M:%S")

    if os.path.exists("logs.csv"):
        df = pd.read_csv("logs.csv")
        render_metrics(df, now_time)
        render_feed(now_time)
        render_detections(df)
        with table_ph.container():
            st.dataframe(
                df.sort_values("Time", ascending=False),
                use_container_width=True, hide_index=True, height=300,
            )
    else:
        render_metrics_empty(now_time)
        render_feed(now_time)
        render_detections_empty()
        with table_ph.container():
            st.markdown(
                '<div class="waiting"><div style="font-size:2rem;opacity:0.3">🗄</div>NO VEHICLE DATA YET — SYSTEM READY</div>',
                unsafe_allow_html=True,
            )

    time.sleep(0.5)
