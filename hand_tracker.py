import cv2
import numpy as np
from collections import deque
from datetime import datetime

class HandTracker:
    def __init__(self, buffer_size=32, skin_lower_hsv=(0, 20, 70), skin_upper_hsv=(20, 255, 255)):
        """
        Initialize hand tracker with HSV skin color detection
        
        Args:
            buffer_size: Size of centroid history for smoothing
            skin_lower_hsv: Lower HSV bound for skin detection
            skin_upper_hsv: Upper HSV bound for skin detection
        """
        self.buffer_size = buffer_size
        self.centroid_history = deque(maxlen=buffer_size)
        self.skin_lower_hsv = np.array(skin_lower_hsv)
        self.skin_upper_hsv = np.array(skin_upper_hsv)
        
        # Virtual object (rectangle) - center at (frame_width/2, frame_height/3)
        self.virtual_object_width = 150
        self.virtual_object_height = 100
        
        # Distance thresholds (in pixels)
        self.warning_distance = 150
        self.danger_distance = 80
        
        # State tracking
        self.current_state = "SAFE"
        self.frame_count = 0
        
    def detect_hand(self, frame):
        """
        Detect hand using HSV skin color segmentation + contour analysis
        Returns: hand centroid (x, y) or None if not detected
        """
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create skin mask
        mask = cv2.inRange(hsv, self.skin_lower_hsv, self.skin_upper_hsv)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, mask
        
        # Get largest contour (likely the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter by area (avoid noise)
        if cv2.contourArea(largest_contour) < 500:
            return None, mask
        
        # Calculate centroid using moments
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), mask
        
        return None, mask
    
    def smooth_centroid(self, centroid):
        """Apply moving average smoothing to centroid"""
        if centroid is None:
            return None
        
        self.centroid_history.append(centroid)
        
        if len(self.centroid_history) > 0:
            avg_x = sum(p[0] for p in self.centroid_history) / len(self.centroid_history)
            avg_y = sum(p[1] for p in self.centroid_history) / len(self.centroid_history)
            return (int(avg_x), int(avg_y))
        
        return centroid
    
    def get_virtual_object_bounds(self, frame_width, frame_height):
        """Get virtual object rectangle bounds"""
        obj_x = frame_width // 2
        obj_y = frame_height // 3
        
        x1 = obj_x - self.virtual_object_width // 2
        y1 = obj_y - self.virtual_object_height // 2
        x2 = obj_x + self.virtual_object_width // 2
        y2 = obj_y + self.virtual_object_height // 2
        
        return (x1, y1, x2, y2), (obj_x, obj_y)
    
    def distance_to_rectangle(self, point, rect_bounds):
        """
        Calculate minimum distance from point to rectangle
        rect_bounds: (x1, y1, x2, y2)
        """
        x, y = point
        x1, y1, x2, y2 = rect_bounds
        
        # Find closest point on rectangle to the given point
        closest_x = max(x1, min(x, x2))
        closest_y = max(y1, min(y, y2))
        
        # Calculate distance
        distance = np.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)
        return distance
    
    def update_state(self, distance):
        """Update state based on distance"""
        if distance < self.danger_distance:
            self.current_state = "DANGER"
        elif distance < self.warning_distance:
            self.current_state = "WARNING"
        else:
            self.current_state = "SAFE"
    
    def process_frame(self, frame):
        """
        Main processing function
        Returns: annotated frame, current state
        """
        self.frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        
        # Detect hand
        centroid, mask = self.detect_hand(frame)
        
        # Smooth centroid
        smoothed_centroid = self.smooth_centroid(centroid)
        
        # Get virtual object bounds
        rect_bounds, obj_center = self.get_virtual_object_bounds(frame_width, frame_height)
        
        # Calculate distance
        distance = float('inf')
        if smoothed_centroid:
            distance = self.distance_to_rectangle(smoothed_centroid, rect_bounds)
            self.update_state(distance)
        else:
            self.current_state = "SAFE"
        
        # Draw virtual object (rectangle)
        x1, y1, x2, y2 = rect_bounds
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "VIRTUAL OBJECT", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw hand centroid
        if smoothed_centroid:
            cx, cy = smoothed_centroid
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 15, (0, 0, 255), 2)
            
            # Draw distance line
            closest_x = max(x1, min(cx, x2))
            closest_y = max(y1, min(cy, y2))
            cv2.line(frame, (cx, cy), (closest_x, closest_y), (255, 0, 0), 1)
            cv2.putText(frame, f"Distance: {distance:.1f}px", (cx + 20, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw state indicator
        state_color = {
            "SAFE": (0, 255, 0),
            "WARNING": (0, 165, 255),
            "DANGER": (0, 0, 255)
        }
        
        color = state_color.get(self.current_state, (0, 255, 0))
        
        # State text - FIXED: Use cv2.FONT_HERSHEY_SIMPLEX instead of FONT_HERSHEY_BOLD
        cv2.rectangle(frame, (10, 10), (250, 70), color, -1)
        cv2.putText(frame, f"STATE: {self.current_state}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # DANGER DANGER warning
        if self.current_state == "DANGER":
            # Blinking effect
            if (self.frame_count // 5) % 2 == 0:
                cv2.putText(frame, "DANGER DANGER", (frame_width // 2 - 150, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 3)
                cv2.putText(frame, "DANGER DANGER", (frame_width // 2 - 150, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 1)
        
        # FPS counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (frame_width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to calibrate skin", (10, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame, self.current_state


def run_tracker(camera_index=0, target_fps=30):
    """
    Run the hand tracking system
    
    Args:
        camera_index: Camera device index (0 for default)
        target_fps: Target frames per second
    """
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    
    # Initialize tracker
    tracker = HandTracker()
    
    print("=== Hand Tracking POC ===")
    print("Instructions:")
    print("  - Keep your hand in front of camera")
    print("  - Approach the green virtual rectangle")
    print("  - Watch for WARNING and DANGER states")
    print("  - Press 'q' to quit")
    print("  - Press 's' to calibrate skin color (place hand in center)")
    print("=" * 40)
    
    fps_history = deque(maxlen=30)
    prev_time = datetime.now()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Flip frame horizontally for selfie view
        frame = cv2.flip(frame, 1)
        
        # Process frame
        annotated_frame, state = tracker.process_frame(frame)
        
        # Calculate FPS
        curr_time = datetime.now()
        delta = (curr_time - prev_time).total_seconds()
        if delta > 0:
            fps = 1.0 / delta
            fps_history.append(fps)
            avg_fps = np.mean(fps_history)
            
            cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        prev_time = curr_time
        
        # Display frame
        cv2.imshow("Hand Tracking POC - Arvyax Assignment", annotated_frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('s'):
            print("Skin calibration mode - place your hand in center of screen")
            print("Press any key to calibrate...")
            while True:
                ret, calib_frame = cap.read()
                if ret:
                    calib_frame = cv2.flip(calib_frame, 1)
                    h, w = calib_frame.shape[:2]
                    
                    # Draw calibration box in center
                    cv2.rectangle(calib_frame, (w//2 - 100, h//2 - 100),
                                (w//2 + 100, h//2 + 100), (0, 255, 0), 2)
                    cv2.putText(calib_frame, "Place hand here", (w//2 - 80, h//2 - 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Calibration", calib_frame)
                    
                    if cv2.waitKey(1) & 0xFF != 255:
                        # Sample skin color from center
                        center_roi = calib_frame[h//2-50:h//2+50, w//2-50:w//2+50]
                        hsv_roi = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
                        
                        # Calculate average HSV values
                        avg_hsv = cv2.mean(hsv_roi)[:3]
                        
                        # Set new bounds (with tolerance)
                        tracker.skin_lower_hsv = np.array([
                            max(0, avg_hsv[0] - 15),
                            max(0, avg_hsv[1] - 40),
                            max(0, avg_hsv[2] - 40)
                        ])
                        tracker.skin_upper_hsv = np.array([
                            min(180, avg_hsv[0] + 15),
                            min(255, avg_hsv[1] + 40),
                            min(255, avg_hsv[2] + 40)
                        ])
                        
                        print(f"Calibrated! New HSV range: {tracker.skin_lower_hsv} - {tracker.skin_upper_hsv}")
                        cv2.destroyWindow("Calibration")
                        break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Program ended.")


if __name__ == "__main__":
    run_tracker()
