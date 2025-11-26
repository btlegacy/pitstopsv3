import streamlit as st
import tempfile
import os
import pandas as pd
import plotly.express as px
from analyzer import PitStopAnalyzer

st.set_page_config(page_title="Pit Stop AI Analyzer", layout="wide")

st.title("🏎️ Pit Stop AI Analyzer")
st.markdown("""
Upload an overhead pit stop video. The system will detect motion in key zones 
(Tires, Fuel, Jack) and generate a performance report.
""")

# Sidebar
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Video (MP4/MOV)", type=["mp4", "mov", "avi"])

# Main Logic
if uploaded_file is not None:
    # Save uploaded file to temp
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Original Video")
        st.video(video_path)

    if st.sidebar.button("Analyze Pit Stop"):
        with st.spinner('Analyzing video frames... this may take a moment.'):
            # Initialize Analyzer
            analyzer = PitStopAnalyzer(video_path)
            
            # Progress bar
            progress_bar = st.progress(0)
            
            # Run Processing
            output_video_path, df = analyzer.process(progress_callback=progress_bar.progress)
            
            st.success("Analysis Complete!")
            
            # --- Results Section ---
            st.divider()
            
            # 1. Timeline Chart (Gantt)
            if not df.empty:
                st.subheader("⏱️ Pit Stop Timeline")
                
                # Create Gantt Chart
                fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Task",
                                  title="Crew Activity Timeline", labels={"Task": "Zone"})
                fig.update_yaxes(autorange="reversed") # FL at top
                # Calculate total duration for x-axis range
                fig.layout.xaxis.type = 'linear'
                fig.data[0].x = df.Duration.tolist() # Fix for plotly timeline numeric data
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. Data Table
                st.subheader("📊 Detailed Timing Stats")
                
                # Highlight critical path (longest duration)
                st.dataframe(df.sort_values(by="Start").style.highlight_max(axis=0, subset=['Duration']))
                
                # Summary Metrics
                c1, c2, c3 = st.columns(3)
                total_time = df['Finish'].max() - df['Start'].min()
                c1.metric("Total Pit Lane Time", f"{total_time:.2f} s")
                c2.metric("Fastest Wheel", f"{df[df['Task'].str.contains('Tire')]['Duration'].min():.2f} s")
                c3.metric("Slowest Wheel", f"{df[df['Task'].str.contains('Tire')]['Duration'].max():.2f} s")

            else:
                st.warning("No significant motion detected in zones. Try a video with more movement.")

            # 3. Processed Video Download
            st.subheader("👁️ Computer Vision Overlay")
            
            # Read the processed file back into memory for download button
            with open(output_video_path, 'rb') as f:
                video_bytes = f.read()
                
            st.download_button(
                label="Download Analyzed Video",
                data=video_bytes,
                file_name="analyzed_pitstop.mp4",
                mime="video/mp4"
            )
            
            # Cleanup temp files
            os.unlink(output_video_path)
    
    # Cleanup input temp file
    # os.unlink(video_path) # Optional: keep for cache
