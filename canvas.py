import os
# Set environment variables for Linux if needed (before importing cv2/Qt)
# May not be necessary on all systems or configurations
# os.environ["QT_QPA_PLATFORM"] = "xcb"
# os.environ["XDG_SESSION_TYPE"] = "x11"
import numpy as np
import cv2
from collections import deque
import mediapipe as mp
import time
import math # For distance calculation

# --- Configuration ---
MAX_POINTS = 1024  # Max points per color deque
DEFAULT_BRUSH_SIZE = 5
DEFAULT_LAG = 5
WEBCAM_ID = 0
INITIAL_SCREEN_RESIZE_FACTOR = 0.8 # How much of screen the paint window initially takes

# --- Initialize MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,            # Process only one hand
    min_detection_confidence=0.5, # Slightly higher confidence
    min_tracking_confidence=0.5  # Slightly higher confidence
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles # For better landmark drawing

# --- Default callback function for trackbars ---
def setValues(x):
    pass

# --- Create the control window ---
cv2.namedWindow("Controls")
cv2.resizeWindow("Controls", 400, 100) # Make controls window smaller
cv2.createTrackbar("Brush Size", "Controls", DEFAULT_BRUSH_SIZE, 25, setValues) # Increased max size
cv2.createTrackbar("Lag/Friction", "Controls", DEFAULT_LAG, 15, setValues) # Renamed & adjusted range

# --- Try to load the webcam ---
cap = cv2.VideoCapture(WEBCAM_ID)
if not cap.isOpened():
    print(f"Error: Could not open webcam ID {WEBCAM_ID}.")
    exit()

# --- Get webcam dimensions ---
webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam resolution: {webcam_width}x{webcam_height}")

# --- Get screen dimensions (for initial sizing) ---
try:
    # This is a common way but might require 'pip install screeninfo'
    from screeninfo import get_monitors
    screen = get_monitors()[0]
    screen_width = screen.width
    screen_height = screen.height
    print(f"Detected screen resolution: {screen_width}x{screen_height}")
except ImportError:
    print("Screeninfo not found ('pip install screeninfo'). Using default 1920x1080 for sizing.")
    screen_width = 1920
    screen_height = 1080

# --- Make paint window large, based on screen size ---
paint_width = int(screen_width * INITIAL_SCREEN_RESIZE_FACTOR)
paint_height = int(screen_height * INITIAL_SCREEN_RESIZE_FACTOR)

# --- Colors and Data Structure ---
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),
          (255, 255, 0), (255, 0, 255), (128, 0, 128), (0, 0, 0),
          (255, 255, 255)] # Added WHITE (ERASER)
color_names = ["BLUE", "GREEN", "RED", "YELLOW", "CYAN", "MAGENTA", "PURPLE", "BLACK", "ERASER"]
# Store deque of normalized (x, y) coordinates or None for breaks
color_points = [deque(maxlen=MAX_POINTS) for _ in range(len(colors))]
color_index = 0 # Start with the first color (BLUE)

# --- Create Canvas ---
paintWindow = np.zeros((paint_height, paint_width, 3)) + 255 # White background
cv2.namedWindow('Paint', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Paint', paint_width, paint_height)

# --- State Variables ---
is_fullscreen = False
is_drawing = False
last_point_norm = None # Store last normalized point for line breaks

# --- Calculate UI dimensions ---
ui_height = 65
button_width = webcam_width // (len(colors) + 1) # +1 for CLEAR ALL button

# --- Smoothing Buffer (using normalized coords now) ---
smoothing_buffer_x = deque(maxlen=20) # Larger buffer for smoother lag
smoothing_buffer_y = deque(maxlen=20)

# --- Calibration Class ---
class Calibration:
    def __init__(self, width, height):
        self.paint_width = width
        self.paint_height = height
        # Store normalized (0-1) boundaries
        self.norm_min_x = 0.15
        self.norm_min_y = 0.15
        self.norm_max_x = 0.85
        self.norm_max_y = 0.85
        self.is_calibrating = False
        self.points_collected = [] # Collect points during calibration
        self.padding = 0.05 # Add padding around detected range

    def start_calibration(self):
        self.is_calibrating = True
        self.points_collected = []
        # Reset to extreme values to capture the first points correctly
        self.norm_min_x = 1.0
        self.norm_min_y = 1.0
        self.norm_max_x = 0.0
        self.norm_max_y = 0.0
        print("Calibration started. Move hand to desired corners. Press 'r' again to finish.")

    def update(self, norm_x, norm_y):
        """Update calibration range if in calibration mode"""
        if not self.is_calibrating:
            return

        self.points_collected.append((norm_x, norm_y))
        self.norm_min_x = min(self.norm_min_x, norm_x)
        self.norm_min_y = min(self.norm_min_y, norm_y)
        self.norm_max_x = max(self.norm_max_x, norm_x)
        self.norm_max_y = max(self.norm_max_y, norm_y)

    def finish_calibration(self):
        self.is_calibrating = False
        if not self.points_collected:
             print("No points collected during calibration. Using default range.")
             self.reset_to_defaults()
             return

        # Add padding
        self.norm_min_x = max(0.0, self.norm_min_x - self.padding)
        self.norm_min_y = max(0.0, self.norm_min_y - self.padding)
        self.norm_max_x = min(1.0, self.norm_max_x + self.padding)
        self.norm_max_y = min(1.0, self.norm_max_y + self.padding)

        # Sanity check: Ensure range is not zero or negative
        if self.norm_max_x <= self.norm_min_x: self.norm_max_x = self.norm_min_x + 0.1
        if self.norm_max_y <= self.norm_min_y: self.norm_max_y = self.norm_min_y + 0.1

        print(f"Calibration finished. Range set (normalized): "
              f"X=[{self.norm_min_x:.2f}, {self.norm_max_x:.2f}], "
              f"Y=[{self.norm_min_y:.2f}, {self.norm_max_y:.2f}]")

    def reset_to_defaults(self):
        self.is_calibrating = False
        self.norm_min_x = 0.15
        self.norm_min_y = 0.15
        self.norm_max_x = 0.85
        self.norm_max_y = 0.85
        print("Calibration reset to defaults.")

    def get_range_x(self):
        return self.norm_max_x - self.norm_min_x

    def get_range_y(self):
        return self.norm_max_y - self.norm_min_y

    def get_calibrated_coords(self, norm_x, norm_y):
        """Maps normalized coords (0-1) to paint window coords using calibration"""
        range_x = self.get_range_x()
        range_y = self.get_range_y()

        # Prevent division by zero and handle edge case where range is tiny
        if range_x < 1e-6 or range_y < 1e-6:
            # Fallback to mapping entire screen if calibration is invalid
            paint_x = int(norm_x * self.paint_width)
            paint_y = int(norm_y * self.paint_height)
        else:
            # Scale and shift based on calibrated range
            paint_x = int(((norm_x - self.norm_min_x) / range_x) * self.paint_width)
            paint_y = int(((norm_y - self.norm_min_y) / range_y) * self.paint_height)

        # Clamp values to be within paint window boundaries
        paint_x = max(0, min(self.paint_width - 1, paint_x))
        paint_y = max(0, min(self.paint_height - 1, paint_y))

        return paint_x, paint_y

    def draw_boundary(self, frame):
        """Draws the calibration boundary on the webcam frame"""
        w = frame.shape[1]
        h = frame.shape[0]
        pt1 = (int(self.norm_min_x * w), int(self.norm_min_y * h))
        pt2 = (int(self.norm_max_x * w), int(self.norm_max_y * h))
        color = (0, 255, 255) # Yellow for boundary
        thickness = 2
        cv2.rectangle(frame, pt1, pt2, color, thickness)
        if self.is_calibrating:
             cv2.putText(frame, "CALIBRATING", (pt1[0], pt1[1] - 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

# Create calibration object
calibration = Calibration(paint_width, paint_height)

# --- Gesture Detection Function ---
def detect_drawing_gesture(hand_landmarks):
    """
    Detects if the hand is in a 'pinch' gesture (index finger tip close to thumb tip).
    Returns True if drawing gesture detected, False otherwise.
    """
    if not hand_landmarks:
        return False

    try:
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP] # For selecting

        # Calculate distance between index and thumb tips (normalized)
        pinch_distance = math.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2 + (index_tip.z - thumb_tip.z)**2)

        # Check if index finger is somewhat straight (distance from wrist > distance from MCP)
        # wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        # index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        # dist_tip_wrist = math.sqrt((index_tip.x - wrist.x)**2 + (index_tip.y - wrist.y)**2)
        # dist_mcp_wrist = math.sqrt((index_mcp.x - wrist.x)**2 + (index_mcp.y - wrist.y)**2)
        # finger_straight = dist_tip_wrist > dist_mcp_wrist # Basic check

        # Simple Pinch: Index and Thumb tips are close
        is_pinching = pinch_distance < 0.1 # Adjust threshold based on testing

        return is_pinching

    except IndexError: # Should not happen with default MediaPipe model
        print("Error accessing hand landmarks.")
        return False
    except Exception as e:
        print(f"Error in gesture detection: {e}")
        return False

# --- Smoothing Function ---
def get_smoothed_point_norm(norm_x, norm_y, lag_factor):
    """Applies smoothing to normalized coordinates for a lag/friction effect."""
    smoothing_buffer_x.append(norm_x)
    smoothing_buffer_y.append(norm_y)

    if len(smoothing_buffer_x) < 2:
        return norm_x, norm_y

    # Simple moving average for lag effect
    avg_x = np.mean(list(smoothing_buffer_x)[-lag_factor:])
    avg_y = np.mean(list(smoothing_buffer_y)[-lag_factor:])

    # More sophisticated weighted average (optional, uncomment to try)
    # alpha = 1.0 - (lag_factor / (len(smoothing_buffer_x) + 1) ) # Weight for current point decreases with lag
    # if len(smoothing_buffer_x) > 1:
    #      prev_avg_x = np.mean(list(smoothing_buffer_x)[:-1])
    #      prev_avg_y = np.mean(list(smoothing_buffer_y)[:-1])
    #      avg_x = alpha * norm_x + (1 - alpha) * prev_avg_x
    #      avg_y = alpha * norm_y + (1 - alpha) * prev_avg_y
    # else:
    #     avg_x = norm_x
    #     avg_y = norm_y

    return avg_x, avg_y


# --- Function to save the drawing ---
def save_drawing(canvas):
    try:
        # Create a 'drawings' subdirectory if it doesn't exist
        if not os.path.exists("drawings"):
            os.makedirs("drawings")

        count = 0
        filename = f"drawings/drawing_{count}.png"
        while os.path.exists(filename):
            count += 1
            filename = f"drawings/drawing_{count}.png"

        # Create a copy to draw the save message on without altering original
        display_canvas = canvas.copy()

        # Draw feedback message on the temporary canvas
        h, w = display_canvas.shape[:2]
        msg = f"Saved as {filename}"
        (text_width, text_height), baseline = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        rect_x = (w - text_width) // 2 - 10
        rect_y = h // 2 - text_height // 2 - 10
        rect_w = text_width + 20
        rect_h = text_height + baseline + 20

        cv2.rectangle(display_canvas, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 200, 0), -1)
        cv2.putText(display_canvas, msg,
                   (rect_x + 10, rect_y + text_height + baseline // 2 + 5), # Center text
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Paint", display_canvas)
        cv2.waitKey(1) # Update display

        # Save the original canvas
        success = cv2.imwrite(filename, canvas)
        if success:
            print(f"Drawing saved successfully as {filename}")
            cv2.waitKey(1500) # Display message for 1.5 seconds
        else:
             print(f"Error: Failed to save drawing as {filename}")
             # Optionally show an error message on screen too
             cv2.waitKey(1500)

    except Exception as e:
        print(f"An error occurred during saving: {e}")
        # Optionally show an error message
        cv2.waitKey(1500)

# --- Function to clear canvas ---
def clear_canvas(paint_window_ref):
    for i in range(len(colors)):
        color_points[i].clear() # Use clear() for deque
        color_points[i].append(None) # Add None to prevent connecting old lines
    paint_window_ref[:] = 255 # Clear the entire canvas
    print("Canvas cleared.")


# --- Main Loop ---
try:
    while True:
        # 1. Read Frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam. Exiting.")
            time.sleep(2) # Pause before exiting
            break

        frame = cv2.flip(frame, 1) # Flip horizontally for intuitive movement
        frame_orig = frame.copy() # Keep original for drawing UI later

        # 2. Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False # Optimize: mark as non-writeable
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True # Make writeable again

        # 3. Get Controls
        brush_size = cv2.getTrackbarPos("Brush Size", "Controls")
        lag_factor = cv2.getTrackbarPos("Lag/Friction", "Controls")
        if brush_size < 1: brush_size = 1
        # Map lag_factor to buffer size for smoothing (adjust multiplier as needed)
        effective_lag = max(1, min(lag_factor + 1, len(smoothing_buffer_x)))


        # 4. Draw UI on Webcam Frame
        ui_frame = frame # Draw UI directly on the main frame
        # Clear All button
        clear_button_rect = (0, 0, button_width, ui_height)
        cv2.rectangle(ui_frame, clear_button_rect[:2], clear_button_rect[2:], (122, 122, 122), -1)
        cv2.putText(ui_frame, "CLEAR", (10, ui_height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Color/Eraser buttons
        for i in range(len(colors)):
            start_x = (i + 1) * button_width
            end_x = (i + 2) * button_width
            if end_x > webcam_width: end_x = webcam_width # Prevent overflow

            button_rect = (start_x, 0, end_x, ui_height)
            color = colors[i]
            cv2.rectangle(ui_frame, button_rect[:2], button_rect[2:], color, -1)

            # Choose text color based on background brightness
            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)

            # Highlight selected button
            if i == color_index:
                 cv2.rectangle(ui_frame, button_rect[:2], button_rect[2:], (255,255,255), 3) # White border
                 text_color = (255, 0, 100) if brightness > 127 else (0, 255, 100) # Contrasting highlight text


            cv2.putText(ui_frame, color_names[i], (start_x + 10, ui_height - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

        # 5. Hand Tracking and Processing
        current_norm_coords = None
        pointer_webcam_coords = None # For drawing cursor on webcam feed
        drawing_gesture_active = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw nice landmarks
                mp_drawing.draw_landmarks(
                    ui_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Get Index Finger Tip coordinates (normalized 0-1)
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                norm_x, norm_y = index_tip.x, index_tip.y

                # Update Calibration if active
                calibration.update(norm_x, norm_y)

                # Get Smoothed Normalized Coordinates for drawing lag
                smoothed_norm_x, smoothed_norm_y = get_smoothed_point_norm(norm_x, norm_y, effective_lag)
                current_norm_coords = (smoothed_norm_x, smoothed_norm_y)

                # Calculate pointer coordinates on webcam feed (using non-smoothed for responsiveness)
                pointer_webcam_coords = (int(norm_x * webcam_width), int(norm_y * webcam_height))

                # Detect Drawing Gesture
                drawing_gesture_active = detect_drawing_gesture(hand_landmarks)

                # Draw Cursor on Webcam Feed
                if pointer_webcam_coords:
                    cursor_color = (0, 255, 0) if drawing_gesture_active else (0, 0, 255) # Green if drawing, Red if not
                    cursor_radius = 10
                    cv2.circle(ui_frame, pointer_webcam_coords, cursor_radius, cursor_color, -1)
                    cv2.circle(ui_frame, pointer_webcam_coords, cursor_radius + 2 , (255,255,255), 1) # White outline

                break # Process only the first detected hand

        # Draw calibration boundary on webcam frame
        calibration.draw_boundary(ui_frame)

        # 6. Handle Interactions (UI clicks, Drawing)
        if pointer_webcam_coords: # Check if a hand was detected this frame
            px, py = pointer_webcam_coords # Pointer position on webcam frame

            # Check UI Button Clicks (only if drawing gesture is active)
            if py < ui_height and drawing_gesture_active:
                time.sleep(0.1) # Small debounce delay

                if clear_button_rect[0] <= px < clear_button_rect[2]: # Clear All
                    clear_canvas(paintWindow)
                else:
                    # Check color/eraser buttons
                    for i in range(len(colors)):
                        start_x = (i + 1) * button_width
                        end_x = (i + 2) * button_width
                        if end_x > webcam_width: end_x = webcam_width
                        if start_x <= px < end_x:
                            color_index = i
                            print(f"Selected: {color_names[color_index]}")
                            break # Found the button

            # Handle Drawing on Canvas
            elif py >= ui_height and current_norm_coords: # Below UI and have smoothed coords
                if drawing_gesture_active:
                    if not is_drawing:
                        # Start of a new line segment
                        is_drawing = True
                        color_points[color_index].appendleft(None) # Add break before starting
                        last_point_norm = None # Reset last point

                    # Add the smoothed normalized point to the deque
                    color_points[color_index].appendleft(current_norm_coords)
                    last_point_norm = current_norm_coords

                else: # Drawing gesture NOT active
                    if is_drawing:
                        # End of a line segment
                        is_drawing = False
                        color_points[color_index].appendleft(None) # Add break
                        last_point_norm = None
            else:
                 # Pointer is outside drawing area (maybe in UI but not clicking)
                 if is_drawing:
                    is_drawing = False
                    color_points[color_index].appendleft(None) # Add break if hand moves to UI while drawing
                    last_point_norm = None

        else: # No hand detected
            if is_drawing:
                 # If hand disappears while drawing, add a break
                 is_drawing = False
                 color_points[color_index].appendleft(None)
                 last_point_norm = None


        # 7. Draw on the Paint Window
        # Clear only the drawing area, keeping info bar intact
        paintWindow[31:, :] = 255 # Clear below info bar

        for i in range(len(colors)):
            points_deque = color_points[i]
            draw_color = colors[i]
            # Iterate through points using pairs for lines
            for j in range(1, len(points_deque)):
                p1_norm = points_deque[j - 1]
                p2_norm = points_deque[j]

                if p1_norm is None or p2_norm is None:
                    continue # Skip breaks

                # Scale normalized points to paint window coordinates using calibration
                p1_scaled = calibration.get_calibrated_coords(p1_norm[0], p1_norm[1])
                p2_scaled = calibration.get_calibrated_coords(p2_norm[0], p2_norm[1])

                # Determine brush size (use larger for eraser)
                current_brush_size = brush_size * 2 if i == color_names.index("ERASER") else brush_size

                # Draw line on paint window
                cv2.line(paintWindow, p1_scaled, p2_scaled, draw_color, max(1, current_brush_size), cv2.LINE_AA) # Use AA for smoother lines


        # 8. Add Info Panel to Paint Window
        info_bar_color = (220, 220, 220)
        cv2.rectangle(paintWindow, (0, 0), (paint_width, 30), info_bar_color, -1)
        info_text = f"Color: {color_names[color_index]} | Brush: {brush_size} | Lag: {lag_factor} | " \
                    f"Keys: (S)ave (C)lear (F)ullscreen (R)ecalibrate (Q)uit"
        cv2.putText(paintWindow, info_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw a border around the paint area (below info bar)
        cv2.rectangle(paintWindow, (0, 31), (paint_width - 1, paint_height - 1), (180, 180, 180), 1)


        # 9. Display Windows
        cv2.imshow("Webcam Feed & UI", ui_frame)
        cv2.imshow("Paint", paintWindow)

        # 10. Handle Keyboard Inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            save_drawing(paintWindow)
            # Redraw info bar after save message potentially clears it
            cv2.rectangle(paintWindow, (0, 0), (paint_width, 30), info_bar_color, -1)
            cv2.putText(paintWindow, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(paintWindow, (0, 31), (paint_width - 1, paint_height - 1), (180, 180, 180), 1)

        elif key == ord("c"):
            clear_canvas(paintWindow)
        elif key == ord("e"): # Select Eraser
             color_index = color_names.index("ERASER")
             print(f"Selected: {color_names[color_index]}")
        elif key == ord("f"): # Toggle fullscreen
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty('Paint', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty('Paint', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Paint', paint_width, paint_height) # Ensure size is restored
        elif key == ord("r"): # Recalibrate
            if calibration.is_calibrating:
                calibration.finish_calibration()
            else:
                calibration.start_calibration()


except Exception as e:
    print(f"\n--- An error occurred during execution ---")
    import traceback
    print(traceback.format_exc())
    print(f"Error message: {e}")
    print("Exiting...")

finally:
    # Release resources
    print("\nReleasing resources...")
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    if 'hands' in locals() and hands: # Check if hands was initialized
        hands.close()
    print("Cleanup complete. Goodbye!")