import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Streamlit UI Setup
st.title("Massage Ball & Pose Detector")
st.write("Step 1: Place the massage ball inside the dashed circle to calibrate color.")
st.write("Step 2: Move with the ball to track body pose.")

# Open webcam
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()  # Placeholder for displaying video frames
calibration_message = st.empty()  # Placeholder for calibration countdown

# Define a region for calibration (center of the screen)
CALIBRATION_CENTER_X = 1000
CALIBRATION_CENTER_Y = 500
CALIBRATION_RADIUS = 400

# Global variable to store ball color range
if "calibrated" not in st.session_state:
    st.session_state.calibrated = False
    st.session_state.calibration_start_time = None
    st.session_state.LOWER_HSV = None
    st.session_state.UPPER_HSV = None
    st.session_state.stop_pressed = False  # Track if "Stop" button is pressed


def calibrate_ball_color(frame):
    """
    Extracts the color of the ball inside the dashed circle (ROI) and calculates HSV color bounds.
    Calibration continues until the color in the ROI is relatively uniform.
    """
    if st.session_state.calibration_start_time is None:
        st.session_state.calibration_start_time = time.time()  # Start timer when calibration begins

    elapsed_time = time.time() - st.session_state.calibration_start_time
    remaining_time = max(0, 3 - int(elapsed_time))  # Ensure it doesn’t go below zero

    # ✅ **Use st.empty() to replace message instead of stacking messages**
    if remaining_time > 0:
        calibration_message.info(f"⏳ Hold the ball steady for {remaining_time} more seconds...")
    
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Extract a circular region in the center
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (CALIBRATION_CENTER_X, CALIBRATION_CENTER_Y), CALIBRATION_RADIUS, 255, -1)
    roi_hsv = hsv[mask == 255]

    # Compute the mean and standard deviation of HSV values
    h_mean, s_mean, v_mean = np.mean(roi_hsv[:, 0]), np.mean(roi_hsv[:, 1]), np.mean(roi_hsv[:, 2])
    h_std, s_std, v_std = np.std(roi_hsv[:, 0]), np.std(roi_hsv[:, 1]), np.std(roi_hsv[:, 2])

    # Define a threshold for uniformity (tune this threshold if necessary)
    uniformity_threshold = 30  # Lower means stricter uniformity check

    if h_std < uniformity_threshold and s_std < uniformity_threshold and v_std < uniformity_threshold:
        # Define HSV range dynamically (±15 for hue, ±40 for saturation/value)
        st.session_state.LOWER_HSV = np.array([max(0, h_mean - 15), max(50, s_mean - 40), max(50, v_mean - 40)], dtype=np.uint8)
        st.session_state.UPPER_HSV = np.array([min(179, h_mean + 15), min(255, s_mean + 40), min(255, v_mean + 40)], dtype=np.uint8)

        st.session_state.calibrated = True
        calibration_message.success("✅ Color Calibrated! Proceeding to detection...")  # ✅ **Replace message instead of appending**
        return True
    else:
        calibration_message.warning("⚠️ Keep the ball steady in the center for better detection...")
        return False

# def detect_massage_ball(frame):
#     """
#     Detects the massage ball using color filtering (HSV thresholding).
#     """
#     if not st.session_state.calibrated:
#         return frame, None  # Skip detection if not calibrated

#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, st.session_state.LOWER_HSV, st.session_state.UPPER_HSV)

#     # Find contours of detected ball
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         max_contour = max(contours, key=cv2.contourArea)
#         ((x, y), radius) = cv2.minEnclosingCircle(max_contour)

#         if radius > 5:
#             cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
#             cv2.putText(frame, f"Ball: ({int(x)}, {int(y)})", (int(x)-50, int(y)-20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             return frame, (int(x), int(y))

#     return frame, None

def detect_massage_ball(frame):
    """
    Detects the massage ball by:
    1. Filtering out objects with high HSV variation (std > 30).
    2. Sorting remaining objects by closest HSV match to the target.
    3. Filtering by circularity to ensure the object is round.
    """

    if not st.session_state.calibrated:
        return frame, None  # Skip detection if not calibrated

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary mask using the calibrated HSV range
    mask = cv2.inRange(hsv, st.session_state.LOWER_HSV, st.session_state.UPPER_HSV)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_objects = []

    for contour in contours:
        # Compute area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Skip small objects
        if area < 100 or perimeter == 0:
            continue  

        # Compute circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Create a mask for the current contour
        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Extract HSV values inside contour
        roi_hsv = hsv[contour_mask == 255]

        # Compute HSV statistics
        h_mean, s_mean, v_mean = np.mean(roi_hsv[:, 0]), np.mean(roi_hsv[:, 1]), np.mean(roi_hsv[:, 2])
        h_std, s_std, v_std = np.std(roi_hsv[:, 0]), np.std(roi_hsv[:, 1]), np.std(roi_hsv[:, 2])
        total_std = h_std + s_std + v_std  # Aggregate HSV variation

        # Step 1: Filter by uniformity (color consistency)
        if total_std > 30:
            continue  # Skip objects with too much color variation

        # Step 2: Compute distance to target color (HSV mean from calibration)
        target_hsv = np.mean([st.session_state.LOWER_HSV, st.session_state.UPPER_HSV], axis=0)
        color_distance = np.linalg.norm(np.array([h_mean, s_mean, v_mean]) - target_hsv)

        # Step 3: Store candidate objects
        candidate_objects.append((contour, color_distance, circularity))

    # Step 4: Sort by closest color match
    candidate_objects.sort(key=lambda x: x[1])  # Sort by color distance (lower is better)

    # Step 5: Select most circular object among top candidates
    for contour, _, circularity in candidate_objects:
        if circularity > 0.6:  # Accept only circular objects
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if radius > 5:  # Ensure minimum size
                # Draw detected ball
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.putText(frame, f"Ball: ({int(x)}, {int(y)})", (int(x)-50, int(y)-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Debugging output
                if "debug_message" not in st.session_state:
                    st.session_state.debug_message = st.empty()
                st.session_state.debug_message.text(f"✅ Selected: Circularity={circularity:.2f}, HSV Distance={color_distance:.2f}")

                return frame, (int(x), int(y))

    return frame, None  # No valid ball detected

def detect_body_pose(frame):
    """
    Detects human body pose using MediaPipe Pose.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
    return frame


# Stop button **(Placed outside the loop to avoid duplication error)**
if st.button("Stop Tracking"):
    st.session_state.stop_pressed = True

while cap.isOpened() and not st.session_state.stop_pressed:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a natural view
    frame = cv2.flip(frame, 1)

    if not st.session_state.calibrated:
        # Draw calibration circle
        cv2.circle(frame, (CALIBRATION_CENTER_X, CALIBRATION_CENTER_Y), CALIBRATION_RADIUS, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Place Ball Here", (CALIBRATION_CENTER_X - 50, CALIBRATION_CENTER_Y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        st.session_state.calibrated = calibrate_ball_color(frame)  # Wait for 3 seconds before calibrating
    else:
        # Detect the ball
        frame, ball_position = detect_massage_ball(frame)

        # Detect the body pose
        frame = detect_body_pose(frame)

    # Convert to RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

cap.release()
cv2.destroyAllWindows()
