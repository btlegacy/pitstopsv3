import cv2
import numpy as np
import pandas as pd
import tempfile
from ultralytics import YOLO

class PitStopAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Load Standard YOLOv8 Nano model (Small, Fast, detects 'person')
        # It will auto-download on first run
        self.model = YOLO('yolov8n.pt') 
        
        self.car_detected = False
        self.zones = {}
        self.activity_log = []
        
        # Tracking duration of activity in zones
        # format: "ZoneName": {active_frames: 0, last_active: False, start_frame: 0}
        self.zone_stats = {}

        # Car Rect for visualization
        self.car_bbox = None 

    def find_car(self, frame):
        """
        Locate the Vasser Sullivan Lexus using HSV Color.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Tuned for Neon Yellow
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([50, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Heavy dilation to merge camo patches
        kernel = np.ones((20, 20), np.uint8) 
        mask = cv2.dilate(mask, kernel, iterations=3)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return None
        
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < (self.width * self.height * 0.05): return None
        
        x, y, w, h = cv2.boundingRect(c)
        return (x, y, w, h)

    def generate_zones(self, car_box):
        """
        Generates zones based on User Description:
        Car moves Left -> Right.
        Top = Outside. Bottom = Inside.
        """
        cx, cy, cw, ch = car_box
        
        # Center of the car
        center_x = cx + cw // 2
        center_y = cy + ch // 2
        
        # Dimensions for zones (Approx 1/3 car length, 1/2 car width)
        zw = int(cw * 0.35) 
        zh = int(ch * 0.5)
        
        # Padding to push zones away from car body
        pad_y = int(ch * 0.2) 

        # Based on Left->Right Movement:
        # Front is RIGHT (Higher X), Rear is LEFT (Lower X)
        # Outside is TOP (Lower Y), Inside is BOTTOM (Higher Y)
        
        zones = {
            # Top Right
            "Outside_Front": [center_x, cy - zh - pad_y, zw, zh],
            
            # Bottom Right
            "Inside_Front":  [center_x, cy + ch + pad_y, zw, zh],
            
            # Top Left
            "Outside_Rear":  [center_x - zw, cy - zh - pad_y, zw, zh],
            
            # Bottom Left
            "Inside_Rear":   [center_x - zw, cy + ch + pad_y, zw, zh],
            
            # Bottom Middle (Fueler)
            "Fueling":       [center_x - zw//2, cy + ch, zw, zh],
            
            # Top Middle (Driver Change)
            "Driver_Change": [center_x - zw//2, cy - zh, zw, zh]
        }
        
        # Initialize stats
        self.zone_stats = {k: {'frames': 0, 'active': False, 'starts': []} for k in zones.keys()}
        
        return zones

    def check_overlap(self, person_box, zone_box):
        """
        Simple AABB Collision detection
        """
        px1, py1, px2, py2 = person_box
        zx, zy, zw, zh = zone_box
        zx2, zy2 = zx + zw, zy + zh
        
        # Check if they intersect
        if (px1 < zx2 and px2 > zx and py1 < zy2 and py2 > zy):
            return True
        return False

    def process(self, progress_callback=None):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = tfile.name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Variables for car locking
        stable_frames = 0
        last_car_pos = None

        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            # 1. Car Detection Phase
            if not self.car_detected:
                car_box = self.find_car(frame)
                
                if car_box:
                    (x, y, w, h) = car_box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                    cv2.putText(frame, "Aligning...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                    
                    if last_car_pos:
                        # Check if car stopped moving
                        dist = abs(x - last_car_pos[0]) + abs(y - last_car_pos[1])
                        if dist < 5: stable_frames += 1
                        else: stable_frames = 0
                    
                    last_car_pos = (x, y, w, h)
                    
                    if stable_frames > 15: # Approx 0.5s stable
                        self.zones = self.generate_zones(last_car_pos)
                        self.car_detected = True
                        self.car_bbox = last_car_pos
            
            # 2. Analysis Phase
            else:
                # Draw Car Anchor
                cx, cy, cw, ch = self.car_bbox
                cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), (255, 255, 0), 1)

                # Run YOLOv8 detection
                # classes=[0] means only detect Class 0 (Person)
                results = self.model.predict(frame, conf=0.3, classes=[0], verbose=False)
                
                current_active_zones = []

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Get person coordinates
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Draw Person
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 1)
                        
                        # Check overlapping zones
                        for z_name, z_box in self.zones.items():
                            if self.check_overlap((x1, y1, x2, y2), z_box):
                                current_active_zones.append(z_name)

                # Update Logic
                for z_name, z_box in self.zones.items():
                    zx, zy, zw, zh = z_box
                    
                    # Determine color based on activity
                    is_active = z_name in current_active_zones
                    
                    if is_active:
                        color = (0, 0, 255) # RED = Working
                        self.zone_stats[z_name]['frames'] += 1
                        if not self.zone_stats[z_name]['active']:
                             self.zone_stats[z_name]['starts'].append(frame_count / self.fps)
                        self.zone_stats[z_name]['active'] = True
                    else:
                        color = (0, 255, 0) # GREEN = Empty
                        self.zone_stats[z_name]['active'] = False
                    
                    # Draw Zone
                    cv2.rectangle(frame, (zx, zy), (zx+zw, zy+zh), color, 2)
                    cv2.putText(frame, z_name, (zx, zy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            out.write(frame)
            frame_count += 1
            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_count / total_frames)

        self.cap.release()
        out.release()
        
        # Compile Report
        final_log = []
        for name, stats in self.zone_stats.items():
            duration = stats['frames'] / self.fps
            if duration > 1.0: # Ignore noise
                final_log.append({
                    "Task": name, 
                    "Total Duration": round(duration, 2),
                    # Simple heuristic: Start time is the first time someone entered the zone
                    "First Activity": round(stats['starts'][0], 2) if stats['starts'] else 0
                })
                
        return output_path, pd.DataFrame(final_log)
