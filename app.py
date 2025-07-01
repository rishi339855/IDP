import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from detector.drowsiness import get_ear
from detector.yawn import is_yawning
from detector.phone_detector import detect_phone
import pandas as pd
from datetime import datetime
import time
import pygame
import threading
import os

# Initialize pygame mixer for sound
pygame.mixer.init()

# Alert sound path - using relative path
alert_path = "alert.wav"

st.title("🚘 Real-Time Driver Monitoring System")

# Initialize session state for event logging and alert tracking
if 'events' not in st.session_state:
    st.session_state.events = []
if 'alert_timers' not in st.session_state:
    st.session_state.alert_timers = {
        'drowsiness': {'start_time': None, 'alert_played': False},
        'yawning': {'start_time': None, 'alert_played': False},
        'phone': {'start_time': None, 'alert_played': False}
    }

# Create columns for alerts
col1, col2, col3 = st.columns(3)

# Initialize alert states
drowsiness_alert = col1.empty()
yawn_alert = col2.empty()
phone_alert = col3.empty()

# Add download button in sidebar
with st.sidebar:
    st.header("📊 Report")
    if st.button("Download Report"):
        if st.session_state.events:
            df = pd.DataFrame(st.session_state.events)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"driver_monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No events recorded yet!")

def play_alarm_for_duration():
    """Play alarm for 3 seconds"""
    if os.path.exists(alert_path):
        try:
            # Load and play the alarm sound
            alarm_sound = pygame.mixer.Sound(alert_path)
            alarm_sound.play()
            
            # Stop the alarm after 3 seconds
            def stop_alarm():
                time.sleep(3)
                alarm_sound.stop()
            
            # Run stop_alarm in a separate thread
            threading.Thread(target=stop_alarm, daemon=True).start()
            
        except Exception as e:
            st.error(f"Error playing alert sound: {str(e)}")

def check_alert_duration(alert_type, is_detected):
    current_time = time.time()
    timer = st.session_state.alert_timers[alert_type]
    
    if is_detected:
        if timer['start_time'] is None:
            timer['start_time'] = current_time
            timer['alert_played'] = False
        # Play alarm immediately when detection starts and then every 4 seconds
        elif not timer['alert_played'] or (current_time - timer['start_time']) >= 4:
            play_alarm_for_duration()
            timer['alert_played'] = True
            timer['start_time'] = current_time  # Reset timer for next alarm
    else:
        timer['start_time'] = None
        timer['alert_played'] = False

run = st.checkbox('Start Camera')

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # Reset alerts
            drowsiness_detected = False
            yawning_detected = False
            phone_detected = False

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])

                    # Drowsiness Detection
                    left_eye = landmarks[LEFT_EYE]
                    right_eye = landmarks[RIGHT_EYE]
                    ear = (get_ear(left_eye) + get_ear(right_eye)) / 2.0
                    if ear < 0.25:
                        drowsiness_detected = True
                        cv2.putText(frame, "DROWSINESS ALERT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        # Log drowsiness event (only if not already logged in the last second)
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        if not st.session_state.events or st.session_state.events[-1]['timestamp'] != current_time:
                            st.session_state.events.append({
                                'timestamp': current_time,
                                'event_type': 'Drowsiness',
                                'ear_value': round(ear, 3)
                            })

                    # Yawn Detection
                    if is_yawning(landmarks):
                        yawning_detected = True
                        cv2.putText(frame, "YAWNING", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                        # Log yawning event (only if not already logged in the last second)
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        if not st.session_state.events or st.session_state.events[-1]['timestamp'] != current_time:
                            st.session_state.events.append({
                                'timestamp': current_time,
                                'event_type': 'Yawning',
                                'details': 'Mouth distance exceeded threshold'
                            })

            # Mobile Phone Detection
            if detect_phone(frame):
                phone_detected = True
                cv2.putText(frame, "MOBILE PHONE DETECTED", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                # Log phone detection event (only if not already logged in the last second)
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if not st.session_state.events or st.session_state.events[-1]['timestamp'] != current_time:
                    st.session_state.events.append({
                        'timestamp': current_time,
                        'event_type': 'Phone Usage',
                        'details': 'Mobile phone detected in frame'
                    })

            # Check alert durations and play sounds if needed
            check_alert_duration('drowsiness', drowsiness_detected)
            check_alert_duration('yawning', yawning_detected)
            check_alert_duration('phone', phone_detected)

            # Update Streamlit alerts
            drowsiness_alert.markdown(f"### {'⚠️ Drowsiness Detected' if drowsiness_detected else '✅ Alert'}")
            yawn_alert.markdown(f"### {'⚠️ Yawning Detected' if yawning_detected else '✅ Alert'}")
            phone_alert.markdown(f"### {'⚠️ Phone Detected' if phone_detected else '✅ Alert'}")

            stframe.image(frame, channels="BGR")

    cap.release()