import streamlit as st
import multiprocessing as mp
import time
import cv2

st.set_page_config(layout="wide")

st.title("AVRS Real-Time Dashboard")

# Create queue
if "frame_queue" not in st.session_state:
    st.session_state.frame_queue = mp.Queue(maxsize=1)

# Start backend process
if "process" not in st.session_state:
    process = mp.Process(
        target=__import__("stream_backend").start_inference,
        args=(st.session_state.frame_queue,)
    )
    process.start()
    st.session_state.process = process

frame_placeholder = st.empty()

# 🔥 LIVE STREAM LOOP (FAST)
while True:
    if not st.session_state.frame_queue.empty():
        frame = st.session_state.frame_queue.get()

        # Convert BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame, use_container_width=True)

    time.sleep(0.03)  # ~30 FPS