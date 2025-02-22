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
CALIBRATION_CENTER_X = 320
CALIBRATION_CENTER_Y = 240
CALIBRATION_RADIUS = 50

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
    """
    if st.session_state.calibration_start_time is None:
        st.session_state.calibration_start_time = time.time()  # Start timer when calibration begins

    elapsed_time = time.time() - st.session_state.calibration_start_time
    remaining_time = max(0, 3 - int(elapsed_time))  # Ensure it doesn’t go below zero

    # ✅ **Use st.empty() to replace message instead of stacking messages**
    if remaining_time > 0:
        calibration_message.info(f"⏳ Hold the ball steady for {remaining_time} more seconds...")
        return False

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Extract a circular region in the center
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (CALIBRATION_CENTER_X, CALIBRATION_CENTER_Y), CALIBRATION_RADIUS, 255, -1)
    roi_hsv = hsv[mask == 255]

    # Compute the mean HSV values for color calibration
    h_mean = int(np.mean(roi_hsv[:, 0]))
    s_mean = int(np.mean(roi_hsv[:, 1]))
    v_mean = int(np.mean(roi_hsv[:, 2]))

    # Define color range dynamically (±15 for hue, ±40 for saturation/value)
    st.session_state.LOWER_HSV = np.array([max(0, h_mean - 15), max(50, s_mean - 40), max(50, v_mean - 40)], dtype=np.uint8)
    st.session_state.UPPER_HSV = np.array([min(179, h_mean + 15), min(255, s_mean + 40), min(255, v_mean + 40)], dtype=np.uint8)

    st.session_state.calibrated = True
    calibration_message.success("✅ Color Calibrated! Proceeding to detection...")  # ✅ **Replace message instead of appending**
    return True


def detect_massage_ball(frame):
    """
    Detects the massage ball using color filtering (HSV thresholding).
    """
    if not st.session_state.calibrated:
        return frame, None  # Skip detection if not calibrated

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, st.session_state.LOWER_HSV, st.session_state.UPPER_HSV)

    # Find contours of detected ball
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(max_contour)

        if radius > 5:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.putText(frame, f"Ball: ({int(x)}, {int(y)})", (int(x)-50, int(y)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return frame, (int(x), int(y))

    return frame, None


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
