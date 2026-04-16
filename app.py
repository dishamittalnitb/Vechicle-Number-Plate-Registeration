import streamlit as st
import pandas as pd
import time
import os
from PIL import Image

st.set_page_config(page_title="AVRS Live Dashboard", page_icon="🚗", layout="wide")

st.title("🚗 AVRS Real-Time Dashboard")
st.markdown("Automated Vehicle Registration System - Live Monitoring")

# ── UI CONTAINERS ──────────────────────────────────────────────────
# We use st.empty() so we can overwrite these sections in a live loop
metrics_placeholder = st.empty()
video_col, log_col = st.columns([6, 7])
with video_col:
    st.subheader("📷 Main Gate (Live Feed)")
    live_feed_placeholder = st.empty()

with log_col:
    st.subheader("Recent Detections")
    recent_logs_placeholder = st.empty()

st.divider()
st.subheader("Vehicle Logs Database")
data_table_placeholder = st.empty()

# ── REAL-TIME LOOP ─────────────────────────────────────────────────
while True:
    # 1. Update Top Metrics & Data Table
    if os.path.exists("logs.csv"):
        df = pd.read_csv("logs.csv")
        
        # Display Metrics
        with metrics_placeholder.container():
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Vehicles Today", len(df))
            m2.metric("Last Gate Activity", "Main Gate")
            m3.metric("System Status", "🟢 Active")
            
        # Display Data Table
        data_table_placeholder.dataframe(df.sort_values(by="Time", ascending=False), use_container_width=True)
        
        # Display Recent Detections (Right Column)
        with recent_logs_placeholder.container():
            # Show the 3 most recent vehicles
            recent_df = df.tail(3).iloc[::-1] 
            for _, row in recent_df.iterrows():
                st.markdown(f"**{row['Plate Number']}**")
                st.caption(f"{row['Time']} | {row['Direction']}")
                # Display the actual cropped image of the plate
                if os.path.exists(row['Image Path']):
                    img = Image.open(row['Image Path'])
                    st.image(img, width=150)
                st.write("---")
    else:
        data_table_placeholder.info("Waiting for first vehicle detection...")

    # 2. Update Live Video Feed (Left Column)
    if os.path.exists("latest_frame.jpg"):
        try:
            # Load and display the latest frame saved by inference.py
            frame = Image.open("latest_frame.jpg")
            live_feed_placeholder.image(frame, channels="BGR", use_container_width=True)
        except Exception:
            pass # Ignore read errors if the file is currently being overwritten

    # Refresh rate (adjust based on your system's performance)
    time.sleep(0.5)