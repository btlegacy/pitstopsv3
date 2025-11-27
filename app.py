import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from analyzer import PitStopAnalyzer

st.set_page_config(page_title="Pit Stop AI - Dynamic", layout="wide")

st.title("🏎️ Dynamic Pit Stop Analyzer")
st.markdown("This system automatically detects the car position to align analysis zones.")

uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Preview
    col1, col2 = st.columns([2, 1])
    with col1:
        st.video(video_path)
    
    with col2:
        st.info("System Logic:")
        st.markdown("""
        1. Scans video for **Neon Yellow/Green** car body.
        2. Waits for car to become **stationary**.
        3. Calculates geometry and **rotates zones** to match car angle.
        4. Measure crew activity in those zones.
        """)

    if st.button("Start Analysis"):
        analyzer = PitStopAnalyzer(video_path)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Calibrating car position...")
        
        output_video_path, df = analyzer.process(progress_callback=progress_bar.progress)
        
        status_text.text("Processing Complete.")
        
        st.divider()
        
        # Results
        if not df.empty:
            st.subheader("⏱️ Pit Stop Timeline")
            
            # Clean up timeline for display
            df['Label'] = df.apply(lambda x: f"{x['Task']} ({x['Duration']:.2f}s)", axis=1)
            
            fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Task", text="Label")
            fig.update_yaxes(autorange="reversed")
            fig.layout.xaxis.type = 'linear'
            fig.data[0].x = df.Duration.tolist()
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats Table
            st.dataframe(df.sort_values(by="Start"))
            
        else:
            st.warning("Analysis finished, but no distinct crew movements were detected after the car stopped.")

        # Download
        with open(output_video_path, 'rb') as f:
            video_bytes = f.read()
            
        st.download_button("Download Analyzed Video", video_bytes, "analyzed_pitstop.mp4", "video/mp4")
        
        os.unlink(output_video_path)
