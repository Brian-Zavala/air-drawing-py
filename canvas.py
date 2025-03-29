import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"
import numpy as np
import cv2
from collections import deque

# Default callback function for trackbars
def setValues(x):
    pass

# Create the HSV trackbar window with wider default range
cv2.namedWindow("Color detectors")
cv2.createTrackbar("Upper Hue", "Color detectors", 180, 180, setValues)  # Max hue
cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, setValues)
cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, setValues)
cv2.createTrackbar("Lower Hue", "Color detectors", 0, 180, setValues)    # Min hue
cv2.createTrackbar("Lower Saturation", "Color detectors", 50, 255, setValues)
cv2.createTrackbar("Lower Value", "Color detectors", 50, 255, setValues)
cv2.createTrackbar("Brush Size", "Color detectors", 3, 15, setValues)    # Increased max brush size

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

# The kernel for dilation 
kernel = np.ones((5, 5), np.uint8)

# Create large canvas
paintWindow = np.zeros((paint_height, paint_width, 3)) + 255
cv2.namedWindow('Paint', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Paint', paint_width, paint_height)

# Flag for fullscreen toggle
is_fullscreen = False

# Calculate UI dimensions
ui_height = 65
button_width = webcam_width // (len(colors) + 1)  # +1 for CLEAR ALL button

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
        self.min_x = float('inf')
        self.min_y = float('inf')
        self.max_x = 0
        self.max_y = 0
        self.calibration_mode = True  # Start in calibration mode
        self.points = []  # Store calibration points
        
    def update(self, x, y, y_threshold):
        """Update calibration with new point if in calibration mode"""
        if not self.calibration_mode or y <= y_threshold:
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
        
        # Ensure we have valid bounds
        if not self.has_data():
            # Set some reasonable defaults if we don't have good data
            width = webcam_width
            height = webcam_height - ui_height
            self.min_x = 0
            self.min_y = ui_height
            self.max_x = width 
            self.max_y = height
            
        print(f"Calibration complete! Tracking area set to: {self.min_x},{self.min_y} to {self.max_x},{self.max_y}")
        
    def reset(self):
        """Reset calibration and start over"""
        self.min_x = float('inf')
        self.min_y = float('inf')
        self.max_x = 0
        self.max_y = 0
        self.calibration_mode = True
        self.points = []
        
    def has_data(self):
        """Check if we have enough calibration data"""
        return (self.max_x > self.min_x + 50 and 
                self.max_y > self.min_y + 50)
                
    def get_x_range(self):
        """Get X range with safety check"""
        return max(self.max_x - self.min_x, 1)  # Prevent division by zero
        
    def get_y_range(self):
        """Get Y range with safety check"""
        return max(self.max_y - self.min_y, 1)  # Prevent division by zero

# Create calibration object
calibration = Calibration()

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
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Getting trackbar positions for HSV values
        u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
        u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
        u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
        l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
        l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
        l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")
        brush_size = cv2.getTrackbarPos("Brush Size", "Color detectors")
        if brush_size < 1:  # Ensure brush size is at least 1
            brush_size = 1
            
        Upper_hsv = np.array([u_hue, u_saturation, u_value])
        Lower_hsv = np.array([l_hue, l_saturation, l_value])

        # Draw UI buttons on the frame
        # Clear All button
        frame = cv2.rectangle(frame, (0, 0), (button_width, ui_height), 
                            (122, 122, 122), -1)
        cv2.putText(frame, "CLEAR ALL", (10, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
                    
        # Add a small guide in the bottom corner to help with calibration
        cv2.putText(frame, "Move to all corners to calibrate", (10, webcam_height-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
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

        # Create a mask for the colored pointer with enhanced processing
        Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
        
        # Apply more aggressive morphological operations to enhance detection
        # First erode to remove small noise
        Mask = cv2.erode(Mask, kernel, iterations=1)
        # Then open to further remove noise
        Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
        # Dilate more aggressively to enlarge the detected area
        Mask = cv2.dilate(Mask, kernel, iterations=2)
        
        # Add text to Mask window to show current HSV values
        mask_info = f"HSV Range: [{l_hue},{l_saturation},{l_value}] to [{u_hue},{u_saturation},{u_value}]"
        cv2.putText(Mask, mask_info, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)

        # Find contours for the pointer
        cnts, _ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        center = None

        # If contours are found
        if len(cnts) > 0:
            # Get the largest contour
            cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            
            # Get the radius of the enclosing circle around the contour
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            
            # Draw the circle around the contour
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            
            # Calculate the center of the contour
            M = cv2.moments(cnt)
            if M['m00'] != 0:  # Prevent division by zero
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                
                # Update calibration with current point
                calibration.update(center[0], center[1], ui_height)
                    
                # Print calibration info occasionally for debugging
                if center[0] % 100 == 0 and center[1] % 100 == 0:
                    print(f"Calibration: X range [{calibration.min_x}-{calibration.max_x}], Y range [{calibration.min_y}-{calibration.max_y}]")
            else:
                center = None

            # Check if user clicked on any UI buttons
            if center and center[1] <= ui_height:
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
            # Add the current center to the active color's points
            # Only add points if they're in the drawing area (below UI)
            elif center and center[1] > ui_height:
                # Store the original webcam coordinates for drawing
                color_points[color_index].appendleft(center)
                
                # Draw a small indicator on the tracking window to show the actual drawing position
                cv2.circle(frame, center, 5, colors[color_index], -1)
        else:
            # If no contour is found, add a new deque for continuous drawing
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
                
                # Don't draw during calibration mode
                if not calibration.calibration_mode:
                    # Use calibration data for scaling
                    scaled_point1 = (
                        int((points[j - 1][0] - calibration.min_x) * paint_width / calibration.get_x_range()),
                        int(((points[j - 1][1] - calibration.min_y) * (paint_height - 30) / calibration.get_y_range()) + 30)
                    )
                    scaled_point2 = (
                        int((points[j][0] - calibration.min_x) * paint_width / calibration.get_x_range()),
                        int(((points[j][1] - calibration.min_y) * (paint_height - 30) / calibration.get_y_range()) + 30)
                    )
                    
                    # Draw on the paint window with scaled brush size
                    scaled_brush_size = int(brush_size * paint_width / webcam_width)
                    if scaled_brush_size < 1:
                        scaled_brush_size = 1
                    cv2.line(paintWindow, scaled_point1, scaled_point2, colors[i], scaled_brush_size)

        # Add a small info panel to the paint window
        info_text = f"Current Color: {color_names[color_index]} | Brush Size: {brush_size} | Press 's' to save, 'c' to clear, 'f' for fullscreen, 'r' to recalibrate, 'q' to quit"
        cv2.rectangle(paintWindow, (0, 0), (paint_width, 30), (220, 220, 220), -1)
        cv2.putText(paintWindow, info_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 1, cv2.LINE_AA)
                    
        # Draw a border to visualize the drawing area
        cv2.rectangle(paintWindow, (0, 30), (paint_width-1, paint_height-1), (200, 200, 200), 1)

        # Display windows
        cv2.imshow("Tracking", frame)
        cv2.imshow("Paint", paintWindow)
        cv2.imshow("Mask", Mask)

        # Display info about calibration mode
        if calibration.calibration_mode:
            # Draw large text indicating calibration mode
            cv2.rectangle(frame, (10, webcam_height//2-40), (webcam_width-10, webcam_height//2+40), (0, 0, 0), -1)
            cv2.putText(frame, "CALIBRATION MODE", (webcam_width//4, webcam_height//2-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Move marker to all 4 corners", (webcam_width//5, webcam_height//2+20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            
            # Draw corner targets to show where to move
            corner_size = 30
            # Top-left corner
            cv2.rectangle(frame, (0, ui_height), (corner_size, ui_height+corner_size), (0, 255, 0), 2)
            # Top-right corner
            cv2.rectangle(frame, (webcam_width-corner_size, ui_height), (webcam_width, ui_height+corner_size), (0, 255, 0), 2)
            # Bottom-left corner
            cv2.rectangle(frame, (0, webcam_height-corner_size), (corner_size, webcam_height), (0, 255, 0), 2)
            # Bottom-right corner
            cv2.rectangle(frame, (webcam_width-corner_size, webcam_height-corner_size), (webcam_width, webcam_height), (0, 255, 0), 2)
            
            # Show instruction to press SPACE when done
            cv2.putText(frame, "Press SPACE when done calibrating", (webcam_width//5, webcam_height-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                      
            # Draw current calibration points
            for point in calibration.points[-20:]:  # Show last 20 points
                cv2.circle(frame, point, 2, (0, 255, 255), -1)
        else:
            # Show the fixed calibration area
            cv2.rectangle(frame, 
                         (calibration.min_x, calibration.min_y), 
                         (calibration.max_x, calibration.max_y), 
                         (0, 255, 0), 1)
            
            # Show calibration info
            cv2.putText(frame, "DRAWING MODE", (webcam_width//3, webcam_height-50),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            calibration_text = f"Tracking: {calibration.min_x},{calibration.min_y} to {calibration.max_x},{calibration.max_y}"
            cv2.putText(frame, calibration_text, (10, webcam_height-30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        
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
            print("Calibration reset. Move your marker across the entire screen to recalibrate.")
        elif key == 32:  # Space bar - finish calibration
            if calibration.calibration_mode:
                calibration.finish_calibration()
                for i in range(len(colors)):
                    color_points[i] = deque(maxlen=1024)  # Clear any points collected during calibration
                paintWindow[:] = 255  # Clear the canvas

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()