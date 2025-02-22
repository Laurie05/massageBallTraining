import cv2
import numpy as np
import mediapipe as mp
import streamlit as st

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Streamlit UI
st.title("Massage Ball & Pose Detector")
st.write("Step 1: Place the massage ball inside the circle to calibrate color.")

# Global color calibration variables
calibrated = False
LOWER_HSV, UPPER_HSV = None, None


def extract_color(frame, roi_x, roi_y, roi_w, roi_h):
    """Extracts the dominant color in HSV format from the selected ROI (ball region)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi_hsv = hsv[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    h_mean = np.mean(roi_hsv[:, :, 0])  # Hue
    s_mean = np.mean(roi_hsv[:, :, 1])  # Saturation
    v_mean = np.mean(roi_hsv[:, :, 2])  # Value

    lower_bound = np.array([max(0, h_mean - 15), max(50, s_mean - 40), max(50, v_mean - 40)])
    upper_bound = np.array([min(179, h_mean + 15), min(255, s_mean + 40), min(255, v_mean + 40)])
    return lower_bound, upper_bound


def detect_massage_ball(frame, lower_hsv, upper_hsv):
    """Detects the massage ball using color-based segmentation and returns its position."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
        if radius > 5:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.putText(frame, f"Ball: ({int(x)}, {int(y)})", (int(x)-50, int(y)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return (int(x), int(y))
    return None


def detect_pose(frame):
    """Detects human body pose using MediaPipe Pose and overlays it on the frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
    return frame


# Webcam capture
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()  # Placeholder for video frame display

# Step 1: Calibration Loop
ROI_X, ROI_Y, ROI_W, ROI_H = 200, 150, 100, 100  # Position of calibration circle
while not calibrated:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Draw dashed circle for ball placement
    cv2.circle(frame, (ROI_X + ROI_W//2, ROI_Y + ROI_H//2), 50, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Place Ball Here", (ROI_X - 30, ROI_Y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    frame_placeholder.image(frame, channels="BGR")

    # Extract color when ball is inside
    LOWER_HSV, UPPER_HSV = extract_color(frame, ROI_X, ROI_Y, ROI_W, ROI_H)
    st.write("✅ Color Calibrated! Proceeding to detection...")
    calibrated = True  # Calibration complete

# Step 2: Ball & Pose Detection
st.write("Step 2: Start moving with the massage ball for detection.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Detect ball
    ball_position = detect_massage_ball(frame, LOWER_HSV, UPPER_HSV)

    # Detect pose
    frame = detect_pose(frame)

    # Display frame
    frame_placeholder.image(frame, channels="BGR")

    if st.button("Stop"):
        break

cap.release()
cv2.destroyAllWindows()
