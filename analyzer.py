import cv2
import numpy as np
import pandas as pd
import tempfile

class PitStopAnalyzer:
    def __init__(self, video_path, sensitivity=25):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # MOG2 Background Subtractor (Detects moving pixels)
        # varThreshold: Lower = More Sensitive
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=sensitivity, detectShadows=False)
        
        self.car_detected = False
        self.car_bbox = None
        self.zones = {}
        
        # Tracking Data
        self.zone_stats = {}
        self.debug_frame = None

    def find_car(self, frame):
        """
        Locates the Vasser Sullivan Neon Yellow Car.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # WIDENED THRESHOLDS: Covers Greenish-Yellow to Orange-Yellow
        lower_yellow = np.array([15, 60, 60]) 
        upper_yellow = np.array([65, 255, 255])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Heavy dilation to merge camo pattern into one blob
        kernel = np.ones((15, 15), np.uint8) 
        mask = cv2.dilate(mask, kernel, iterations=4)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return None
        
        # Get largest blob
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        
        # Must be at least 5% of the screen
        if area < (self.width * self.height * 0.05): return None
        
        x, y, w, h = cv2.boundingRect(c)
        return (x, y, w, h)

    def generate_zones(self, car_box):
        cx, cy, cw, ch = car_box
        
        # Center of Car
        center_x = cx + cw // 2
        center_y = cy + ch // 2
        
        # Box Sizes (Tuned for GT3 shape)
        zw = int(cw * 0.3) # Zone Width
        zh = int(ch * 0.6) # Zone Height
        
        # Spacing (Push boxes away from car body)
        pad_y = int(ch * 0.25)
        
        # ZONES CONFIGURATION
        # Based on: Car moving Left->Right. Top=Outside. Bottom=Inside.
        zones = {
            # TIRES
            "Outside_Front": [center_x + int(cw*0.2), cy - zh - pad_y, zw, zh],
            "Inside_Front":  [center_x + int(cw*0.2), cy + ch + pad_y, zw, zh],
            "Outside_Rear":  [center_x - int(cw*0.3), cy - zh - pad_y, zw, zh],
            "Inside_Rear":   [center_x - int(cw*0.3), cy + ch + pad_y, zw, zh],
            
            # SERVICES
            "Driver_Change": [center_x - int(cw*0.1), cy - zh - pad_y, zw, zh], # Top Middle
            "Fueling":       [center_x - int(cw*0.2), cy + ch, int(cw*0.2), int(zh*0.8)], # Bot Mid
            "Fire_Bottle":   [center_x - int(cw*0.5), cy + ch + pad_y, int(zw*0.8), int(zh*0.8)] # Behind Fueler
        }
        
        # Init stats
        self.zone_stats = {k: {'active': False, 'start_f': 0, 'end_f': 0, 'total_f': 0} for k in zones}
        return zones

    def process(self, progress_callback=None):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = tfile.name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Latching: How many frames a zone stays active after motion stops
        # 30 frames = ~1 second (Prevents flickering when fueler holds still)
        patience = 30 
        cooldowns = {} 

        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            # --- PHASE 1: FIND CAR ---
            if not self.car_detected:
                car_box = self.find_car(frame)
                if car_box:
                    # Draw Blue Box to show we found it
                    (x,y,w,h) = car_box
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 200, 0), 3)
                    cv2.putText(frame, "LOCKED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,200,0), 3)
                    
                    # Lock immediately for now (assuming car is stopped)
                    # In production, check velocity here
                    self.car_bbox = car_box
                    self.zones = self.generate_zones(car_box)
                    self.car_detected = True
                    cooldowns = {k: 0 for k in self.zones}
                else:
                    cv2.putText(frame, "SEARCHING FOR CAR...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            # --- PHASE 2: TRACK CREW ---
            else:
                # 1. Motion Mask
                fgmask = self.fgbg.apply(frame)
                # Remove noise
                _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                
                # Draw Car Anchor (Blue)
                cx, cy, cw, ch = self.car_bbox
                cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), (255, 200, 0), 2)

                # 2. Check Zones
                for name, rect in self.zones.items():
                    zx, zy, zw, zh = rect
                    
                    # Boundary Checks
                    zx, zy = max(0, zx), max(0, zy)
                    zw, zh = min(zw, self.width-zx), min(zh, self.height-zy)
                    
                    roi = thresh[zy:zy+zh, zx:zx+zw]
                    if roi.size == 0: continue

                    # Calculate Motion Percentage
                    white_pixels = np.count_nonzero(roi)
                    total_pixels = zw * zh
                    ratio = white_pixels / total_pixels
                    
                    # Threshold: 5% of the zone must be moving to trigger
                    is_moving = ratio > 0.05 
                    
                    # Latching Logic
                    if is_moving:
                        cooldowns[name] = patience
                        if not self.zone_stats[name]['active']:
                            self.zone_stats[name]['active'] = True
                            self.zone_stats[name]['start_f'] = frame_count
                    elif cooldowns[name] > 0:
                        cooldowns[name] -= 1
                    else:
                        if self.zone_stats[name]['active']:
                            # Activity Ended
                            self.zone_stats[name]['active'] = False
                            duration = frame_count - self.zone_stats[name]['start_f']
                            # Only log if activity lasted > 1.5 seconds (ignore drive-by)
                            if duration > (self.fps * 1.5):
                                self.zone_stats[name]['total_f'] += duration

                    # Visuals
                    color = (0, 0, 255) if cooldowns[name] > 0 else (0, 255, 0)
                    thick = 3 if cooldowns[name] > 0 else 1
                    
                    cv2.rectangle(frame, (zx, zy), (zx+zw, zy+zh), color, thick)
                    cv2.putText(frame, name, (zx, zy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            out.write(frame)
            frame_count += 1
            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_count / total_frames)

        self.cap.release()
        out.release()
        
        # Compile Report
        final_log = []
        for name, stats in self.zone_stats.items():
            duration_sec = stats['total_f'] / self.fps
            if duration_sec > 0:
                final_log.append({
                    "Task": name,
                    "Duration": round(duration_sec, 2)
                })
        
        return output_path, pd.DataFrame(final_log)
