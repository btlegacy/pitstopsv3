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
        
        # Background subtractor
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        
        # State tracking
        self.car_detected = False
        self.zones = {} # Will be populated dynamically
        self.activity_log = []
        self.state = {}

    def find_car_alignment(self, frame):
        """
        Detects the car based on neon yellow/green color and determines orientation.
        Returns: (center_x, center_y), (width, height), angle, box_points
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # HSV range for the "Highlighter Yellow" (Vasser Sullivan / Aston Martin colors)
        lower_yellow = np.array([25, 50, 50])
        upper_yellow = np.array([45, 255, 255])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Clean up mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (the car body)
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        
        # Filter small noise
        if area < 5000: 
            return None
            
        # Get Rotated Bounding Box
        rect = cv2.minAreaRect(c)
        (center, size, angle) = rect
        
        # Normalizing angle logic for OpenCV versions
        if size[0] < size[1]:
            angle = angle + 90
            size = (size[1], size[0]) # Swap to make width always the "long" side

        return center, size, angle, cv2.boxPoints(rect)

    def generate_dynamic_zones(self, center, size, angle):
        """
        Creates zones relative to the car's center and rotation.
        """
        cx, cy = center
        w, h = size # w is length of car, h is width of car
        
        # Geometric Multipliers (Tuned to typical GT3 Car proportions)
        offsets = {
            "FL_Tire":  (-0.35, -0.60), # Front Left
            "FR_Tire":  (-0.35,  0.60), # Front Right
            "RL_Tire":  ( 0.35, -0.60), # Rear Left
            "RR_Tire":  ( 0.35,  0.60), # Rear Right
            "Fuel_Rig": ( 0.10, -0.40), # Mid-rear
            "Jack":     ( 0.55,  0.00)  # Rear Jack
        }

        zones = {}
        
        # Convert angle to radians
        rad = np.radians(angle)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)

        for name, (off_x, off_y) in offsets.items():
            # Calculate unrotated offset
            dx = w * off_x
            dy = h * off_y
            
            # Rotate the offset
            new_dx = dx * cos_a - dy * sin_a
            new_dy = dx * sin_a + dy * cos_a
            
            # New center
            zone_cx = int(cx + new_dx)
            zone_cy = int(cy + new_dy)
            
            # Define box size (approx 120x120 pixels, scaled by car size)
            box_s = int(h * 0.4) 
            
            # Store as [x, y, w, h] for OpenCV
            zones[name] = [zone_cx - box_s//2, zone_cy - box_s//2, box_s, box_s]
            
        return zones

    def process(self, progress_callback=None):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = tfile.name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calibration Phase variables
        frames_stable = 0
        last_center = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_time = frame_count / self.fps
            
            # --- Phase 1: Dynamic Calibration (Wait for Car to Stop) ---
            if not self.car_detected:
                # Try to find car
                result = self.find_car_alignment(frame)
                
                if result:
                    center, size, angle, box_points = result
                    
                    # --- FIX APPLIED HERE: np.int0 -> np.int32 ---
                    box = np.int32(box_points) 
                    cv2.drawContours(frame, [box], 0, (255, 255, 0), 2)
                    
                    # Check if stationary
                    if last_center:
                        movement = np.linalg.norm(np.array(center) - np.array(last_center))
                        if movement < 2.0: # Moving less than 2 pixels per frame
                            frames_stable += 1
                        else:
                            frames_stable = 0
                    
                    last_center = center
                    
                    # If stable for 0.5 seconds, LOCK THE ZONES
                    if frames_stable > (self.fps * 0.5):
                        self.zones = self.generate_dynamic_zones(center, size, angle)
                        self.car_detected = True
                        # Initialize state tracking for these new zones
                        self.state = {k: {'active': False, 'start_time': 0, 'last_seen': 0} for k in self.zones}
                        
            # --- Phase 2: Analysis (Once car is locked) ---
            else:
                # 1. Motion Detection
                fgmask = self.fgbg.apply(frame)
                _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
                
                # 2. Check Dynamic Zones
                for name, (x, y, w, h) in self.zones.items():
                    # Ensure roi is within frame bounds
                    x, y = max(0, x), max(0, y)
                    
                    roi = thresh[y:y+h, x:x+w]
                    if roi.size == 0: continue

                    white_pixels = np.sum(roi == 255)
                    motion_ratio = white_pixels / (w * h)
                    
                    is_moving = motion_ratio > 0.15 # 15% threshold
                    
                    # Logic
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

                    # Draw Zone
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            out.write(frame)
            frame_count += 1
            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_count / total_frames)

        self.cap.release()
        out.release()
        
        return output_path, pd.DataFrame(self.activity_log)
