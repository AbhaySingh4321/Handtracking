# Handtracking
Project Overview
A real-time hand tracking system that detects hand position and triggers state-based warnings when approaching a virtual object on screen. Built using classical computer vision techniques without pose detection APIs.
🛠️ Installation
Prerequisites
Python 3.8+
Webcam/Camera device

# Setup
Clone or download the project
cd hand-tracking
# Create virtual environment (recommended)
python -m venv venv
Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
# Install dependencies
pip install opencv-python numpy
# Verify installation
python -c "import cv2; import numpy; print('✓ All dependencies installed')"
🚀 Running the Application
python hand_tracker.py
Expected output:

=== Hand Tracking POC ===
Instructions:
  - Keep your hand in front of camera
  - Approach the green virtual rectangle
  - Watch for WARNING and DANGER states
  - Press 'q' to quit
  - Press 's' to calibrate skin color
========================================
📱 User Interface & Controls
Visual Feedback
Element	         |Meaning
____________________________________________________|
Green Rectangle	 |Virtual object/boundary           |
Red Dot	         |Your hand position (centroid)     |
Blue Line	       |Distance from hand to object      |
Green State Box	 |SAFE state                        |
Orange State Box |WARNING state (approaching)       |
Red State Box	   |DANGER state (too close)          |
"DANGER DANGER"	 |Blinking alert when hand too close|

Skin Calibration (Important!)
Press s during runtime
Place your hand in the center calibration box
Press any key to sample your skin color
System automatically adjusts HSV range for better tracking
🔍 How It Works
1. Hand Detection
Converts camera frame from BGR to HSV color space
Applies HSV thresholding for skin color detection
Uses morphological operations (dilate/erode) to clean noise
Finds contours and extracts largest region (hand)
Calculates centroid using image moments
Skin color HSV range (customizable via calibration)
skin_lower_hsv = (0, 20, 70)
skin_upper_hsv = (20, 255, 255)
2. Centroid Smoothing
Maintains history of 32 recent centroids
Applies moving average filter
Reduces jitter and improves tracking stability
Critical for reliable distance calculation
3. Virtual Object
Green rectangle positioned at 1/3 from top center
Dimensions: 150×100 pixels
Acts as the boundary to track proximity
4. Distance Calculation
Computes minimum distance from hand centroid to rectangle boundary
Distance = perpendicular distance to nearest edge or corner
5. State Logic
if distance < 80px → DANGER (red alert)
elif distance < 150px → WARNING (orange)
else → SAFE (green)
📊 Performance
Benchmarks (640×480 resolution, CPU-only)
Metric	        |Value
---------------------------------------
FPS (avg)	      |18-22 FPS
Latency	        |~45-55ms per frame
CPU Usage	      |~35-45% (single core)
Memory	        |~150-200 MB
⚙️ Technical Stack
OpenCV 4.x
NumPy
HSV Skin Segmentation + Contour Analysis
Moving Average Filter
Euclidean Distance
Simple threshold-based logic
