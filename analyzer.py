import cv2
import numpy as np
import pandas as pd
import tempfile

class PitStopAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.car_detected = False
        self.zones = {}
        self.activity_log = []
        self.state = {}
        
        # Debug: Store the detected car rect for visualization
        self.car_rect_points = None

    def find_car_alignment(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # HSV for Vasser Sullivan Yellow
        # If detection fails, try widening this range
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([50, 255, 255])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # --- FIX 1: HEAVY DILATION ---
        # The camo pattern breaks the car into pieces. 
        # We dilate (smear) the mask to connect them into one big blob.
        kernel = np.ones((20, 20), np.uint8) 
        mask = cv2.dilate(mask, kernel, iterations=3)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        
        # Threshold: Must be a significant portion of the screen (e.g. > 5% of pixels)
        if area < (self.width * self.height * 0.05): 
            return None
            
        rect = cv2.minAreaRect(c)
        (center, size, angle) = rect
        (w, h) = size

        # --- FIX 2: ORIENTATION LOGIC ---
        # Ensure 'Width' is the long side (Length of car)
        if w < h:
            w, h = h, w
            angle += 90
            
        return center, (w, h), angle, cv2.boxPoints(rect)

    def generate_dynamic_zones(self, center, size, angle):
        cx, cy = center
        w, h = size # w = Car Length, h = Car Width
        
        # --- FIX 3: WIDER OFFSETS ---
        # 0.0 is center. 0.5 is edge of car body.
        # We need > 0.5 to target the tires/mechanics.
        offsets = {
            "FL_Tire":  (-0.35, -0.75), # Moved from -0.60 to -0.75
            "FR_Tire":  (-0.35,  0.75), 
            "RL_Tire":  ( 0.35, -0.75), 
            "RR_Tire":  ( 0.35,  0.75), 
            "Fuel_Rig": ( 0.15, -0.75), 
            "Jack_R":   ( 0.60,  0.00)  
        }

        zones = {}
        rad = np.radians(angle)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)

        for name, (off_x, off_y) in offsets.items():
            # Apply offsets to car dimensions
            dx = w * off_x
            dy = h * off_y
            
            # Rotate
            new_dx = dx * cos_a - dy * sin_a
            new_dy = dx * sin_a + dy * cos_a
            
            zone_cx = int(cx + new_dx)
            zone_cy = int(cy + new_dy)
            
            # --- FIX 4: LARGER BOXES ---
            # Box size relative to car width
            box_s = int(h * 0.6) 
            
            zones[name] = [zone_cx - box_s//2, zone_cy - box_s//2, box_s, box_s]
            
        return zones

    def process(self, progress_callback=None):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = tfile.name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_stable = 0
        last_center = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_time = frame_count / self.fps
            
            if not self.car_detected:
                result = self.find_car_alignment(frame)
                
                if result:
                    center, size, angle, box_points = result
                    
                    # Store for drawing
                    self.car_rect_points = np.int32(box_points)
                    
                    # Check stability
                    if last_center:
                        movement = np.linalg.norm(np.array(center) - np.array(last_center))
                        if movement < 3.0: 
                            frames_stable += 1
                        else:
                            frames_stable = 0
                    last_center = center
                    
                    # Lock on faster (0.3s)
                    if frames_stable > (self.fps * 0.3):
                        self.zones = self.generate_dynamic_zones(center, size, angle)
                        self.car_detected = True
                        self.state = {k: {'active': False, 'start_time': 0, 'last_seen': 0} for k in self.zones}
                
                # Draw Debug Car Box (Cyan)
                if self.car_rect_points is not None:
                    cv2.drawContours(frame, [self.car_rect_points], 0, (255, 255, 0), 3)
                    cv2.putText(frame, "DETECTING CAR...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            else:
                # Normal processing
                fgmask = self.fgbg.apply(frame)
                _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
                
                # Draw Debug Car Box (Cyan) to show anchor
                if self.car_rect_points is not None:
                    cv2.drawContours(frame, [self.car_rect_points], 0, (255, 255, 0), 2)

                for name, (x, y, w, h) in self.zones.items():
                    # Clamp coordinates
                    x, y = max(0, x), max(0, y)
                    roi = thresh[y:y+h, x:x+w]
                    
                    is_moving = False
                    if roi.size > 0:
                        white_pixels = np.sum(roi == 255)
                        motion_ratio = white_pixels / (w * h)
                        is_moving = motion_ratio > 0.15

                    color = (0, 255, 0)
                    if is_moving:
                        self.state[name]['last_seen'] = current_time
                        if not self.state[name]['active']:
                            self.state[name]['active'] = True
                            self.state[name]['start_time'] = current_time
                    
                    if self.state[name]['active']:
                        color = (0, 0, 255)
                        if (current_time - self.state[name]['last_seen']) > 0.5:
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
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            out.write(frame)
            frame_count += 1
            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_count / total_frames)

        self.cap.release()
        out.release()
        return output_path, pd.DataFrame(self.activity_log)
