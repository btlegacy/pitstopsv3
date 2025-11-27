import streamlit as st
import tempfile
import os
import pandas as pd
import plotly.express as px
from analyzer import PitStopAnalyzer

st.set_page_config(page_title="Endurance Pit AI", layout="wide")

st.title("🏎️ Endurance Pit Stop Analyzer")
st.markdown("""
**Configuration:**
- **Flow:** Left to Right
- **Roles:** Driver Change (Top), Fuel (Bottom Mid), Tires (Corners)
- **Tech:** YOLOv8 Human Detection + Color Anchoring
""")

uploaded_file = st.sidebar.file_uploader("Upload Pit Stop Video", type=["mp4", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    if st.button("Run Analysis"):
        st.info("Downloading AI Model & Processing... (First run takes 30s)")
        
        analyzer = PitStopAnalyzer(video_path)
        progress_bar = st.progress(0)
        
        output_path, df = analyzer.process(progress_callback=progress_bar.progress)
        
        st.success("Processing Complete")
        
        # Video
        with open(output_path, 'rb') as f:
            st.download_button("Download Overlay Video", f.read(), "analyzed_pit.mp4")
            
        # Stats
        if not df.empty:
            st.subheader("⏱️ Crew Performance")
            
            # Sort chronologically by who started working first
            df = df.sort_values("First Activity")
            
            # Bar Chart for Duration
            fig = px.bar(df, x="Total Duration", y="Task", orientation='h', 
                         title="Time Spent in Work Zone", color="Task")
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df)
            
            # Insight logic based on user description
            st.markdown("### 🧠 AI Insights")
            
            try:
                fuel_time = df.loc[df['Task'] == 'Fueling', 'Total Duration'].values[0]
                st.write(f"**Fuel Time:** {fuel_time}s")
            except:
                st.write("Fueler not detected consistently.")
                
            try:
                # Compare Front vs Rear Tire speed
                of_time = df.loc[df['Task'] == 'Outside_Front', 'Total Duration'].values[0]
                or_time = df.loc[df['Task'] == 'Outside_Rear', 'Total Duration'].values[0]
                
                if of_time > or_time:
                    st.write(f"⚠️ **Outside Front** took {of_time - or_time:.2f}s longer than Outside Rear.")
                else:
                    st.write(f"✅ **Outside Front** was faster than Rear.")
            except:
                pass
                
        else:
            st.warning("No crew activity detected. Check video lighting or camera angle.")
            
        os.unlink(output_path)
