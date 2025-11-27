import cv2
import numpy as np
import pandas as pd
import tempfile

class PitStopAnalyzer:
    def __init__(self, video_path, zones):
        """
        zones: Dictionary of zones with coordinates scaled to video resolution
               Format: {'FL_Tire': [x, y, w, h], ...}
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Use the zones provided by the user
        self.zones = zones
        
        # Background subtractor
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

        # State tracking
        self.activity_log = []
        self.state = {k: {'active': False, 'start_time': 0, 'last_seen': 0} for k in self.zones}

    def process(self, progress_callback=None):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = tfile.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
            for name, rect in self.zones.items():
                x, y, w, h = rect
                
                # Safety check for boundary conditions
                if x < 0: x = 0
                if y < 0: y = 0
                if x+w > self.width: w = self.width - x
                if y+h > self.height: h = self.height - y
                
                roi = thresh[y:y+h, x:x+w]
                
                # Handle empty ROI edge case
                if roi.size == 0:
                    continue

                white_pixels = np.sum(roi == 255)
                total_pixels = w * h
                
                if total_pixels == 0: continue
                
                motion_ratio = white_pixels / total_pixels
                is_moving = motion_ratio > 0.10
                
                # Logic & Visuals
                color = (0, 255, 0)
                
                if is_moving:
                    self.state[name]['last_seen'] = current_time
                    if not self.state[name]['active']:
                        self.state[name]['active'] = True
                        self.state[name]['start_time'] = current_time
                
                if self.state[name]['active']:
                    color = (0, 0, 255)
                    time_since_motion = current_time - self.state[name]['last_seen']
                    
                    if time_since_motion > patience:
                        duration = self.state[name]['last_seen'] - self.state[name]['start_time']
                        if duration > 1.5: 
                            self.activity_log.append({
                                "Task": name,
                                "Start": self.state[name]['start_time'],
                                "Finish": self.state[name]['last_seen'],
                                "Duration": duration
                            })
                        self.state[name]['active'] = False

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            out.write(frame)
            frame_count += 1
            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_count / total_frames)

        self.cap.release()
        out.release()
        
        return output_path, pd.DataFrame(self.activity_log)
