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
st.write("Webcam-based ball tracking and pose detection.")

# Open webcam
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()  # Placeholder for displaying video frames

# Massage ball color detection settings
calibrated = False
LOWER_HSV, UPPER_HSV = np.array([0, 120, 120]), np.array([10, 255, 255])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for natural view
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect ball using color filter
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
        if radius > 5:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.putText(frame, f"Ball: ({int(x)}, {int(y)})", (int(x)-50, int(y)-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Detect pose
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

    # Convert frame to display in Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

    if st.button("Stop"):
        break

cap.release()
cv2.destroyAllWindows()
