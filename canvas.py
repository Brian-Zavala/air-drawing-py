import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"
import numpy as np
import cv2
from collections import deque
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Default callback function for trackbars
def setValues(x):
    pass

# Create the control window
cv2.namedWindow("Controls")
cv2.createTrackbar("Brush Size", "Controls", 3, 15, setValues)
cv2.createTrackbar("Smoothing", "Controls", 5, 10, setValues)  # Added smoothing control

# Try to load the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get webcam dimensions
webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Use webcam dimensions as a base for the paint window, but make it significantly larger
# This will help ensure the drawing area covers most of the screen
screen_width = 1920  # Standard monitor width - adjust if needed
screen_height = 1080  # Standard monitor height - adjust if needed

# Make paint window fill most of a standard screen
paint_width = int(screen_width * 0.9)
paint_height = int(screen_height * 0.9)

# Color points arrays with deque to store the drawing points
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), 
          (255, 255, 0), (255, 0, 255), (128, 0, 128), (0, 0, 0)]  # Added more colors
color_names = ["BLUE", "GREEN", "RED", "YELLOW", "CYAN", "MAGENTA", "PURPLE", "BLACK"]
color_points = [deque(maxlen=1024) for _ in range(len(colors))]
color_index = 0

# Create large canvas
paintWindow = np.zeros((paint_height, paint_width, 3)) + 255
cv2.namedWindow('Paint', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Paint', paint_width, paint_height)

# Flag for fullscreen toggle
is_fullscreen = False

# Calculate UI dimensions
ui_height = 65
button_width = webcam_width // (len(colors) + 1)  # +1 for CLEAR ALL button

# Create a smoothing buffer for drawing points (for friction effect)
smoothing_buffer_x = deque(maxlen=10)
smoothing_buffer_y = deque(maxlen=10)

# Last point for distance calculation
last_point = None
last_draw_time = time.time()
is_drawing = False

# Function to save the drawing
def save_drawing(paintWindow):
    filename = f"drawing_{len([f for f in os.listdir('.') if f.startswith('drawing_')])}.png"
    cv2.imwrite(filename, paintWindow)
    print(f"Drawing saved as {filename}")
    # Also display a message on the paint window
    temp = paintWindow.copy()
    cv2.rectangle(temp, (paint_width//3, paint_height//2-30), 
                 (2*paint_width//3, paint_height//2+30), (0, 200, 0), -1)
    cv2.putText(temp, f"Saved as {filename}", 
               (paint_width//3 + 20, paint_height//2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Paint", temp)
    cv2.waitKey(1000)  # Display the message for 1 second
    cv2.imshow("Paint", paintWindow)

# Add auto-calibration feature to track min/max coordinates
class Calibration:
    def __init__(self):
        self.min_x = 0.2  # Start with some reasonable defaults (20% from edge)
        self.min_y = 0.2
        self.max_x = 0.8  # 80% of width
        self.max_y = 0.8
        self.calibration_mode = False
        self.points = []
        
    def update(self, x, y):
        """Update calibration with new point if in calibration mode"""
        if not self.calibration_mode:
            return
            
        # Store the point for later processing
        self.points.append((x, y))
        
        # Update boundaries
        self.min_x = min(self.min_x, x)
        self.min_y = min(self.min_y, y)
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)
        
    def finish_calibration(self):
        """Complete calibration and lock in the boundaries"""
        self.calibration_mode = False
        print(f"Calibration complete! Tracking area set to: {self.min_x},{self.min_y} to {self.max_x},{self.max_y}")
        
    def reset(self):
        """Reset calibration and start over"""
        self.min_x = 0.2
        self.min_y = 0.2
        self.max_x = 0.8
        self.max_y = 0.8
        self.calibration_mode = True
        self.points = []
                
    def get_x_range(self):
        """Get X range with safety check"""
        return max(self.max_x - self.min_x, 0.1)  # Prevent division by zero
        
    def get_y_range(self):
        """Get Y range with safety check"""
        return max(self.max_y - self.min_y, 0.1)  # Prevent division by zero

# Create calibration object using normalized coordinates (0-1) instead of pixels
calibration = Calibration()

# Function to detect hand gesture (open/closed)
def detect_drawing_gesture(hand_landmarks):
    """
    Detect if the hand is in a drawing gesture
    Uses distance between index fingertip and thumb tip
    """
    if not hand_landmarks:
        return False
    
    # Get coordinates of index finger tip and thumb tip
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    
    # Calculate distance between fingertips
    distance = np.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
    
    # If distance is small enough, consider it a pinch/drawing gesture
    return distance < 0.1  # Threshold can be adjusted

# Function to get smoothed drawing point
def get_smoothed_point(x, y, smoothing_factor):
    """Apply smoothing to make drawing have 'friction'"""
    # Add point to buffer
    smoothing_buffer_x.append(x)
    smoothing_buffer_y.append(y)
    
    # Calculate average (smoothed) position
    if len(smoothing_buffer_x) < 2:
        return x, y
    
    # Apply weighted average based on smoothing factor
    # Higher smoothing = more previous points influence = more lag/friction
    weight_current = 1 - (smoothing_factor / 10)
    weight_prev = smoothing_factor / 10
    
    x_smooth = x * weight_current + np.mean(list(smoothing_buffer_x)[:-1]) * weight_prev
    y_smooth = y * weight_current + np.mean(list(smoothing_buffer_y)[:-1]) * weight_prev
    
    return int(x_smooth), int(y_smooth)

# Main loop
try:
    while True:
        # Reading the frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break
        
        # Flipping the frame to see same side of yours
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Hands
        results = hands.process(rgb_frame)
        
        # Get current controls
        brush_size = cv2.getTrackbarPos("Brush Size", "Controls")
        smoothing_factor = cv2.getTrackbarPos("Smoothing", "Controls")
        if brush_size < 1:
            brush_size = 1
            
        # Draw UI buttons on the frame
        # Clear All button
        frame = cv2.rectangle(frame, (0, 0), (button_width, ui_height), 
                            (122, 122, 122), -1)
        cv2.putText(frame, "CLEAR ALL", (10, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        
        # Color buttons
        for i in range(len(colors)):
            start_x = (i + 1) * button_width
            end_x = (i + 2) * button_width
            if end_x > webcam_width:
                end_x = webcam_width
                
            frame = cv2.rectangle(frame, (start_x, 0), (end_x, ui_height),
                                colors[i], -1)
            # Choose text color based on background color brightness
            text_color = (255, 255, 255) if sum(colors[i]) < 500 else (0, 0, 0)
            cv2.putText(frame, color_names[i], (start_x + 10, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        text_color, 2, cv2.LINE_AA)
        
        # Hand landmark processing
        center = None
        drawing_gesture = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks for visual feedback
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
                
                # Check for drawing gesture
                drawing_gesture = detect_drawing_gesture(hand_landmarks)
                
                # Get index finger tip as pointer
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Convert normalized coordinates to pixel values
                x, y = int(index_tip.x * webcam_width), int(index_tip.y * webcam_height)
                
                # Draw circle at index finger tip
                circle_color = (0, 255, 0) if drawing_gesture else (0, 0, 255)
                cv2.circle(frame, (x, y), 10, circle_color, -1)
                
                # Update center point
                center = (x, y)
                
                # Update calibration with current point
                calibration.update(index_tip.x, index_tip.y)  # Using normalized coordinates
                
                # Add gesture status text
                gesture_text = "DRAWING" if drawing_gesture else "NOT DRAWING"
                cv2.putText(frame, gesture_text, (x - 50, y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, circle_color, 2, cv2.LINE_AA)
        
        # Check if user clicked on any UI buttons
        if center and center[1] <= ui_height and drawing_gesture:
            if 0 <= center[0] <= button_width:  # Clear All button
                for i in range(len(colors)):
                    color_points[i] = deque(maxlen=1024)
                paintWindow[:] = 255  # Clear the entire canvas
            else:
                # Check which color button was pressed
                for i in range(len(colors)):
                    start_x = (i + 1) * button_width
                    end_x = (i + 2) * button_width
                    if end_x > webcam_width:
                        end_x = webcam_width
                        
                    if start_x <= center[0] <= end_x:
                        color_index = i
                        break
        
        # Process drawing based on gesture
        if center and center[1] > ui_height:
            # Apply smoothing for "friction" effect
            smoothed_x, smoothed_y = get_smoothed_point(center[0], center[1], smoothing_factor)
            smoothed_center = (smoothed_x, smoothed_y)
            
            if drawing_gesture:
                if not is_drawing:
                    # Just started drawing
                    is_drawing = True
                    # Reset last point to avoid jumps
                    last_point = smoothed_center
                
                # Only add the point if we're drawing
                color_points[color_index].appendleft(smoothed_center)
                
                # Draw the point on the webcam frame
                cv2.circle(frame, smoothed_center, 5, colors[color_index], -1)
            else:
                if is_drawing:
                    # Just stopped drawing
                    is_drawing = False
                
                # Add None to create a break in the line
                color_points[color_index].appendleft(None)
        else:
            # If no hand is detected, add None to create a break in the line
            for i in range(len(colors)):
                color_points[i].appendleft(None)
        
        # Draw lines of all colors on the canvas and frame
        for i in range(len(colors)):
            points = list(color_points[i])
            for j in range(1, len(points)):
                if points[j - 1] is None or points[j] is None:
                    continue
                
                # Draw on the webcam frame
                cv2.line(frame, points[j - 1], points[j], colors[i], brush_size)
                
                # Scale the points to the paint window size
                scaled_point1 = (
                    int(points[j - 1][0] * paint_width / webcam_width),
                    int(points[j - 1][1] * paint_height / webcam_height)
                )
                scaled_point2 = (
                    int(points[j][0] * paint_width / webcam_width),
                    int(points[j][1] * paint_height / webcam_height)
                )
                
                # Draw on the paint window with scaled brush size
                scaled_brush_size = int(brush_size * 1.5)  # Make brush slightly bigger on canvas
                if scaled_brush_size < 1:
                    scaled_brush_size = 1
                cv2.line(paintWindow, scaled_point1, scaled_point2, colors[i], scaled_brush_size)

        # Add a small info panel to the paint window
        info_text = f"Current Color: {color_names[color_index]} | Brush Size: {brush_size} | Smoothing: {smoothing_factor} | Press 's' to save, 'c' to clear, 'f' for fullscreen, 'r' to recalibrate, 'q' to quit"
        cv2.rectangle(paintWindow, (0, 0), (paint_width, 30), (220, 220, 220), -1)
        cv2.putText(paintWindow, info_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 1, cv2.LINE_AA)
                    
        # Draw a border to visualize the drawing area
        cv2.rectangle(paintWindow, (0, 30), (paint_width-1, paint_height-1), (200, 200, 200), 1)

        # Display windows
        cv2.imshow("Tracking", frame)
        cv2.imshow("Paint", paintWindow)
        
        # Handle keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Quit
            break
        elif key == ord("s"):  # Save the drawing
            save_drawing(paintWindow)
        elif key == ord("c"):  # Clear the drawing
            for i in range(len(colors)):
                color_points[i] = deque(maxlen=1024)
            paintWindow[:] = 255  # Clear the entire canvas
            # Redraw the info panel
            cv2.rectangle(paintWindow, (0, 0), (paint_width, 30), (220, 220, 220), -1)
            cv2.putText(paintWindow, info_text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 1, cv2.LINE_AA)
        elif key == ord("f"):  # Toggle fullscreen mode
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty('Paint', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty('Paint', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Paint', paint_width, paint_height)
        elif key == ord("r"):  # Reset calibration
            calibration.reset()
            print("Calibration reset. Move your hand across the entire screen to recalibrate.")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()