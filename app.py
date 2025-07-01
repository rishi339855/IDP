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
import streamlit_authenticator as stauth
from db import (
    get_user, create_user, update_user, get_all_drivers, get_all_managers,
    get_unassigned_drivers, assign_driver_to_manager, get_drivers_for_manager,
    log_ride, get_rides_for_driver, get_all_rides, log_trip, get_trips_for_driver
)
from bson import ObjectId
from fpdf import FPDF

# Initialize pygame mixer for sound
pygame.mixer.init()

# Alert sound path - using relative path
alert_path = "alert.wav"

# --- PDF GENERATION ---
def generate_trip_pdf(trip, events):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Driver Monitoring Trip Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Driver: {trip['driver']}", ln=True)
    pdf.cell(200, 10, txt=f"Start Point: {trip['start_point']}", ln=True)
    pdf.cell(200, 10, txt=f"Destination: {trip['destination']}", ln=True)
    pdf.cell(200, 10, txt=f"Start Time: {trip['start_time']}", ln=True)
    if 'end_time' in trip:
        pdf.cell(200, 10, txt=f"End Time: {trip['end_time']}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Monitoring Events:", ln=True)
    pdf.ln(5)
    if not events:
        pdf.cell(200, 10, txt="No events recorded.", ln=True)
    else:
        for event in events:
            line = f"[{event.get('timestamp', '')}] {event.get('event_type', '')}"
            if 'details' in event:
                line += f" - {event['details']}"
            if 'ear_value' in event:
                line += f" (EAR: {event['ear_value']})"
            pdf.multi_cell(0, 10, txt=line)
    return pdf.output(dest='S').encode('latin1')

# --- NAVIGATION STACK ---
if 'nav_stack' not in st.session_state:
    st.session_state.nav_stack = ['home']
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

def go_to(page):
    st.session_state.nav_stack.append(page)
    st.session_state.current_page = page

def go_back():
    if len(st.session_state.nav_stack) > 1:
        st.session_state.nav_stack.pop()
    st.session_state.current_page = st.session_state.nav_stack[-1]

# --- REGISTRATION ---
def register_user():
    st.subheader('Register')
    new_username = st.text_input('Choose a username')
    new_password = st.text_input('Choose a password', type='password')
    role = st.selectbox('Role', ['driver', 'manager'])
    if st.button('Register'):
        if get_user(new_username):
            st.error('Username already exists!')
        elif not new_username or not new_password:
            st.error('Username and password required!')
        else:
            hashed_pw = stauth.Hasher.hash(new_password)
            user_obj = {'username': new_username, 'password': hashed_pw, 'role': role}
            if role == 'driver':
                user_obj['fleet_manager'] = None
            create_user(user_obj)
            st.success('Registration successful! You can now log in.')
            go_back()
    if st.button('Back'):
        go_back()

# --- LOGIN SYSTEM ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None

if not st.session_state.logged_in:
    if st.session_state.current_page == 'home':
        # --- BEAUTIFUL LOGIN PAGE HEADER ---
        st.markdown(
            """
            <div style='text-align: center; margin-top: 30px;'>
                <h1 style='color: #2E86C1; font-size: 2.8rem; margin-bottom: 0.2em;'>Driver Drowsiness & Distraction Monitoring System</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.subheader('Login')
        login_card = st.container()
        with login_card:
            st.markdown(
                """
                <style>
                div[data-testid="stTextInput"] label, div[data-testid="stPassword"] label {
                    font-weight: 600;
                }
                div[data-testid="stTextInput"] input, div[data-testid="stPassword"] input {
                    background: #2E86C1;
                    border-radius: 8px;
                    border: 1px solid #D5D8DC;
                    padding: 0.5em;
                    color: #fff !important;
                }
                div[data-testid="stTextInput"] input::placeholder, div[data-testid="stPassword"] input::placeholder {
                    color: #fff !important;
                    opacity: 1;
                }
                button[kind="primary"] {
                    background: linear-gradient(90deg, #2E86C1 60%);
                    color: white;
                    border-radius: 8px;
                    font-weight: 700;
                    font-size: 1.1rem;
                    margin-top: 1em;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            username = st.text_input('Username')
            password = st.text_input('Password', type='password')
            if st.button('Login'):
                user = get_user(username)
                if user and stauth.Hasher.check_pw(password, user['password']):
                    st.session_state.logged_in = True
                    st.session_state.role = user['role']
                    st.session_state.username = username
                    st.session_state.nav_stack = ['dashboard']
                    st.session_state.current_page = 'dashboard'
                    st.rerun()
                else:
                    st.error('Invalid credentials')
            st.write("Don't have an account?")
            if st.button('Register here'):
                go_to('register')
        st.stop()
    elif st.session_state.current_page == 'register':
        register_user()
        st.stop()

# --- END LOGIN SYSTEM ---

# --- MAIN APP NAVIGATION ---
if st.session_state.current_page == 'dashboard':
    if st.session_state.role == 'driver':
        st.title("🚘 Real-Time Driver Monitoring System")
        if st.button('Back'):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.session_state.username = None
            st.session_state.nav_stack = ['home']
            st.session_state.current_page = 'home'
            st.rerun()

        # --- SIDEBAR NAVIGATION FOR DRIVER ---
        with st.sidebar:
            st.header("Driver Menu")
            driver_option = st.radio(
                "Select an option:",
                ["Start Monitoring", "Lane Change Detection", "Download Report"],
                key="driver_sidebar_option"
            )

        if driver_option == "Start Monitoring":
            # Trip selection UI
            st.subheader("Trip Details")
            start_point = st.text_input("Start Point")
            destination = st.text_input("Destination")
            trip_started = st.session_state.get('trip_started', False)
            if not trip_started:
                if st.button("Start Monitoring"):
                    if not start_point or not destination:
                        st.warning("Please enter both start point and destination.")
                    else:
                        trip = {
                            'driver': st.session_state.username,
                            'start_point': start_point,
                            'destination': destination,
                            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        trip_id = log_trip(trip)
                        st.session_state.trip_started = True
                        st.session_state.current_trip_id = trip_id
                        st.success(f"Trip started from {start_point} to {destination}!")
                        st.rerun()
            else:
                # Fetch current trip details
                trips = get_trips_for_driver(st.session_state.username)
                current_trip = None
                for t in trips:
                    if str(t['_id']) == st.session_state.current_trip_id:
                        current_trip = t
                        break
                st.info(f"Trip in progress: {current_trip['start_point']} → {current_trip['destination']}")
                # --- ALERT FUNCTIONS ---
                def play_alarm_for_duration():
                    if os.path.exists(alert_path):
                        try:
                            alarm_sound = pygame.mixer.Sound(alert_path)
                            alarm_sound.play()
                            def stop_alarm():
                                time.sleep(3)
                                alarm_sound.stop()
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
                        elif not timer['alert_played'] or (current_time - timer['start_time']) >= 4:
                            play_alarm_for_duration()
                            timer['alert_played'] = True
                            timer['start_time'] = current_time
                    else:
                        timer['start_time'] = None
                        timer['alert_played'] = False
                # Monitoring UI
                if 'alert_timers' not in st.session_state:
                    st.session_state.alert_timers = {
                        'drowsiness': {'start_time': None, 'alert_played': False},
                        'yawning': {'start_time': None, 'alert_played': False},
                        'phone': {'start_time': None, 'alert_played': False}
                    }
                col1, col2, col3 = st.columns(3)
                drowsiness_alert = col1.empty()
                yawn_alert = col2.empty()
                phone_alert = col3.empty()
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
                            drowsiness_detected = False
                            yawning_detected = False
                            phone_detected = False
                            if results.multi_face_landmarks:
                                for face_landmarks in results.multi_face_landmarks:
                                    h, w, _ = frame.shape
                                    landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
                                    left_eye = landmarks[LEFT_EYE]
                                    right_eye = landmarks[RIGHT_EYE]
                                    ear = (get_ear(left_eye) + get_ear(right_eye)) / 2.0
                                    if ear < 0.25:
                                        drowsiness_detected = True
                                        cv2.putText(frame, "DROWSINESS ALERT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                        log_ride({
                                            'timestamp': current_time,
                                            'event_type': 'Drowsiness',
                                            'ear_value': round(ear, 3),
                                            'driver': st.session_state.username,
                                            'trip_id': st.session_state.current_trip_id
                                        })
                                    if is_yawning(landmarks):
                                        yawning_detected = True
                                        cv2.putText(frame, "YAWNING", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                        log_ride({
                                            'timestamp': current_time,
                                            'event_type': 'Yawning',
                                            'details': 'Mouth distance exceeded threshold',
                                            'driver': st.session_state.username,
                                            'trip_id': st.session_state.current_trip_id
                                        })
                            if detect_phone(frame):
                                phone_detected = True
                                cv2.putText(frame, "MOBILE PHONE DETECTED", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                log_ride({
                                    'timestamp': current_time,
                                    'event_type': 'Phone Usage',
                                    'details': 'Mobile phone detected in frame',
                                    'driver': st.session_state.username,
                                    'trip_id': st.session_state.current_trip_id
                                })
                            check_alert_duration('drowsiness', drowsiness_detected)
                            check_alert_duration('yawning', yawning_detected)
                            check_alert_duration('phone', phone_detected)
                            drowsiness_alert.markdown(f"### {'⚠️ Drowsiness Detected' if drowsiness_detected else '✅ Alert'}")
                            yawn_alert.markdown(f"### {'⚠️ Yawning Detected' if yawning_detected else '✅ Alert'}")
                            phone_alert.markdown(f"### {'⚠️ Phone Detected' if phone_detected else '✅ Alert'}")
                            stframe.image(frame, channels="BGR")
                    cap.release()
                if st.button('End Trip'):
                    # Mark trip as ended
                    from pymongo import MongoClient
                    client = MongoClient("mongodb://localhost:27017/IDP")
                    db = client["IDP"]
                    db["trips"].update_one({'_id': ObjectId(st.session_state.current_trip_id)}, {"$set": {"end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}})
                    st.session_state.trip_started = False
                    st.session_state.current_trip_id = None
                    st.success("Trip ended!")
                    st.rerun()
            if not st.session_state.get('trip_started', False) and st.session_state.get('current_trip_id'):
                # Show trip summary and download PDF
                trips = get_trips_for_driver(st.session_state.username)
                trip = None
                for t in trips:
                    if str(t['_id']) == st.session_state.current_trip_id:
                        trip = t
                        break
                if trip:
                    st.subheader("Trip Summary")
                    st.write(f"Start: {trip['start_point']}")
                    st.write(f"Destination: {trip['destination']}")
                    st.write(f"Start Time: {trip['start_time']}")
                    st.write(f"End Time: {trip.get('end_time', 'N/A')}")
                    # Get events for this trip
                    all_events = get_rides_for_driver(st.session_state.username)
                    trip_events = [e for e in all_events if e.get('trip_id') == st.session_state.current_trip_id]
                    pdf_bytes = generate_trip_pdf(trip, trip_events)
                    st.download_button(
                        label="Download Trip Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"trip_report_{trip['start_point']}_to_{trip['destination']}.pdf",
                        mime="application/pdf"
                    )
        elif driver_option == "Lane Change Detection":
            st.subheader("Lane Change Detection")
            st.info("Lane change detection feature coming soon!")
        elif driver_option == "Download Report":
            st.subheader("Download Report")
            trips = get_trips_for_driver(st.session_state.username)
            if not trips:
                st.warning("No trips recorded yet!")
            else:
                for trip in trips:
                    st.markdown(f"**Trip:** {trip['start_point']} → {trip['destination']} | Start: {trip['start_time']} | End: {trip.get('end_time', 'N/A')}")
                    all_events = get_rides_for_driver(st.session_state.username)
                    trip_events = [e for e in all_events if e.get('trip_id') == str(trip['_id'])]
                    pdf_bytes = generate_trip_pdf(trip, trip_events)
                    st.download_button(
                        label=f"Download Trip Report (PDF) [{trip['start_point']} → {trip['destination']}]",
                        data=pdf_bytes,
                        file_name=f"trip_report_{trip['start_point']}_to_{trip['destination']}.pdf",
                        mime="application/pdf",
                        key=f"driver_download_pdf_{trip['_id']}"
                    )
    elif st.session_state.role == 'manager':
        st.title("📊 Fleet Manager Dashboard")
        if st.button('Back'):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.session_state.username = None
            st.session_state.nav_stack = ['home']
            st.session_state.current_page = 'home'
            st.rerun()
        manager_username = st.session_state.username
        all_drivers = get_all_drivers()
        unassigned_drivers = [d['username'] for d in get_unassigned_drivers()]
        my_drivers = [d['username'] for d in get_drivers_for_manager(manager_username)]
        st.subheader('Driver Management')
        section = st.radio('Select an action:', ['Show & Assign Unassigned Drivers', 'Show My Drivers'], key='manager_section')
        if section == 'Show & Assign Unassigned Drivers':
            st.markdown('**Unassigned Drivers**')
            if unassigned_drivers:
                st.info(f"Unassigned Drivers: {', '.join(unassigned_drivers)}")
                selected_driver = st.radio('Select a driver to assign to yourself:', unassigned_drivers, key='assign_driver_radio')
                if st.button('Assign Selected Driver', key='assign_selected_driver_btn'):
                    if selected_driver:
                        assign_driver_to_manager(selected_driver, manager_username)
                        st.success(f"Driver '{selected_driver}' assigned to you!")
                        st.rerun()
                    else:
                        st.warning('Please select a driver to assign.')
            else:
                st.success("All drivers are assigned to a fleet manager.")
        elif section == 'Show My Drivers':
            st.markdown('**Drivers Under You**')
            if my_drivers:
                for drv in my_drivers:
                    if st.button(f"View Dashboard: {drv}", key=f"view_{drv}"):
                        go_to(f'driver_dashboard_{drv}')
                        st.rerun()
            else:
                st.info('No drivers assigned to you yet.')
        all_rides = get_all_rides()
        st.header("Driver Event Logs")
        if all_rides:
            df = pd.DataFrame(all_rides)
            st.dataframe(df)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name=f"fleet_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No events recorded yet!")
        st.stop()
# --- DRIVER DASHBOARD FOR MANAGER ---
if st.session_state.current_page.startswith('driver_dashboard_'):
    driver_username = st.session_state.current_page.replace('driver_dashboard_', '')
    st.markdown(f"### Dashboard for Driver: {driver_username}")
    if st.button('Back'):
        go_back()
        st.rerun()
    # Show all trips for this driver
    trips = get_trips_for_driver(driver_username)
    if not trips:
        st.info('No trips recorded for this driver.')
    else:
        for trip in trips:
            st.markdown(f"**Trip:** {trip['start_point']} → {trip['destination']} | Start: {trip['start_time']} | End: {trip.get('end_time', 'N/A')}")
            # Get events for this trip
            all_events = get_rides_for_driver(driver_username)
            trip_events = [e for e in all_events if e.get('trip_id') == str(trip['_id'])]
            pdf_bytes = generate_trip_pdf(trip, trip_events)
            st.download_button(
                label=f"Download Trip Report (PDF) [{trip['start_point']} → {trip['destination']}]",
                data=pdf_bytes,
                file_name=f"trip_report_{trip['start_point']}_to_{trip['destination']}.pdf",
                mime="application/pdf",
                key=f"download_pdf_{trip['_id']}"
            )
    st.stop()