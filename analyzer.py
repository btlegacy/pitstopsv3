import cv2
import numpy as np
import pandas as pd
import tempfile
import os

class PitStopAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Background subtractor for motion detection
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

        # Define Zones relative to 1920x1080 (normalized later)
        # Based on your screenshot provided
        base_w, base_h = 1920, 1080
        self.zones = {
            "FL_Tire":  [480, 550, 200, 200],  # x, y, w, h
            "FR_Tire":  [480, 250, 200, 200],
            "RL_Tire":  [1250, 550, 200, 200],
            "RR_Tire":  [1250, 250, 200, 200],
            "Fuel_Rig": [850, 250, 150, 150],
            "Jack_R":   [1500, 400, 150, 150]
        }
        
        # Scale zones to actual video resolution
        scale_x = self.width / base_w
        scale_y = self.height / base_h
        for k, v in self.zones.items():
            self.zones[k] = [int(v[0]*scale_x), int(v[1]*scale_y), int(v[2]*scale_x), int(v[3]*scale_y)]

        # Statistics storage
        self.activity_log = []
        # Track current state: {zone: {'active': bool, 'start_time': float, 'last_seen': float}}
        self.state = {k: {'active': False, 'start_time': 0, 'last_seen': 0} for k in self.zones}

    def process(self, progress_callback=None):
        # Create a temp file for the output video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = tfile.name
        
        # Codec for MP4 (H.264 is best for web, but mp4v is safer for local OpenCV generation)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Debounce threshold (seconds): how long motion must stop before we consider the task "done"
        patience = 0.5 

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_time = frame_count / self.fps
            
            # 1. Motion Detection
            fgmask = self.fgbg.apply(frame)
            _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # 2. Check Zones
            for name, (x, y, w, h) in self.zones.items():
                roi = thresh[y:y+h, x:x+w]
                white_pixels = np.sum(roi == 255)
                total_pixels = w * h
                motion_ratio = white_pixels / total_pixels
                
                # Threshold: >10% of zone moving = Active
                is_moving = motion_ratio > 0.10
                
                # Visuals
                color = (0, 255, 0) # Green (Idle)
                
                # State Machine Logic
                if is_moving:
                    self.state[name]['last_seen'] = current_time
                    if not self.state[name]['active']:
                        self.state[name]['active'] = True
                        self.state[name]['start_time'] = current_time
                
                # Check for completion (Debounce)
                if self.state[name]['active']:
                    color = (0, 0, 255) # Red (Active)
                    time_since_motion = current_time - self.state[name]['last_seen']
                    
                    if time_since_motion > patience:
                        # Event finished
                        duration = self.state[name]['last_seen'] - self.state[name]['start_time']
                        if duration > 1.0: # Filter noise < 1 second
                            self.activity_log.append({
                                "Task": name,
                                "Start": self.state[name]['start_time'],
                                "Finish": self.state[name]['last_seen'],
                                "Duration": duration
                            })
                        self.state[name]['active'] = False

                # Draw Zone
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            out.write(frame)
            frame_count += 1
            
            # Update Streamlit progress bar
            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_count / total_frames)

        self.cap.release()
        out.release()
        
        return output_path, pd.DataFrame(self.activity_log)
