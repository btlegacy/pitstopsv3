import streamlit as st
import tempfile
import os
import pandas as pd
import plotly.express as px
from analyzer import PitStopAnalyzer

st.set_page_config(page_title="Endurance Pit AI", layout="wide")

st.title("🏎️ Endurance Pit Stop Analyzer (Motion)")
st.markdown("""
**Robust Mode:** Uses color anchoring to find the car, then monitors pixel motion in the calculated crew zones.
""")

# Sidebar Config
st.sidebar.header("Settings")
sensitivity = st.sidebar.slider("Motion Sensitivity", 10, 100, 25, 
                                help="Lower = Detects smaller movements. Higher = Ignores noise.")

uploaded_file = st.sidebar.file_uploader("Upload Pit Stop Video", type=["mp4", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col1, col2 = st.columns([2, 1])
    with col1:
        st.video(video_path)

    if st.sidebar.button("Run Analysis"):
        status_text = st.empty()
        status_text.info("Initializing Analysis...")
        
        progress_bar = st.progress(0)
        
        # Run Analysis
        analyzer = PitStopAnalyzer(video_path, sensitivity=sensitivity)
        output_path, df = analyzer.process(progress_callback=progress_bar.progress)
        
        status_text.success("Analysis Complete!")
        
        # --- Results ---
        st.divider()
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("👁️ AI Overlay")
            with open(output_path, 'rb') as f:
                st.download_button("Download Video", f.read(), "analyzed_pit.mp4")
            st.info("Check this video to see if the boxes aligned correctly.")
            
        with c2:
            st.subheader("📊 Crew Stats")
            if not df.empty:
                df = df.sort_values("Duration", ascending=False)
                
                # Chart
                fig = px.bar(df, x="Duration", y="Task", orientation='h', 
                             title="Activity Duration per Zone", color="Task")
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No sustained activity detected. Try lowering the Sensitivity slider.")

        os.unlink(output_path)
