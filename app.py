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
    
    # Set up colors (RGB values)
    pdf.set_fill_color(102, 126, 234)  # Primary blue
    pdf.set_text_color(255, 255, 255)  # White text
    
    # Header with gradient-like effect
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, txt="Driver Monitoring Trip Report", ln=True, align='C', fill=True)
    
    # Reset colors for content
    pdf.set_fill_color(245, 245, 245)  # Light gray background
    pdf.set_text_color(51, 51, 51)     # Dark gray text
    
    # Trip information section with colored background
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_fill_color(240, 248, 255)  # Alice blue background
    pdf.cell(0, 10, txt="Trip Information", ln=True, fill=True)
    pdf.ln(5)
    
    # Trip details with alternating row colors
    pdf.set_font("Arial", '', 11)
    details = [
        ("Driver", trip['driver']),
        ("Start Point", trip['start_point']),
        ("Destination", trip['destination']),
        ("Start Time", trip['start_time'])
    ]
    
    if 'end_time' in trip:
        details.append(("End Time", trip['end_time']))
    
    for i, (label, value) in enumerate(details):
        # Alternate row colors
        if i % 2 == 0:
            pdf.set_fill_color(248, 250, 252)  # Very light blue
        else:
            pdf.set_fill_color(255, 255, 255)  # White
        
        pdf.cell(50, 8, txt=f"{label}:", ln=0, fill=True)
        pdf.cell(0, 8, txt=value, ln=True, fill=True)
    
    # Events section
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_fill_color(255, 193, 7)  # Warning yellow background
    pdf.set_text_color(51, 51, 51)   # Dark text
    pdf.cell(0, 10, txt="Monitoring Events", ln=True, fill=True)
    pdf.ln(5)
    
    if not events:
        pdf.set_font("Arial", '', 11)
        pdf.set_fill_color(240, 248, 255)  # Light blue background
        pdf.cell(0, 8, txt="No events recorded - Safe driving!", ln=True, fill=True)
    else:
        # Event type color mapping
        event_colors = {
            'Drowsiness': (255, 99, 71),    # Tomato red
            'Yawning': (255, 165, 0),       # Orange
            'Phone Usage': (220, 20, 60),   # Crimson
            'Lane Change': (138, 43, 226),  # Blue violet
            'Speed': (255, 215, 0)          # Gold
        }
        
        pdf.set_font("Arial", '', 10)
        
        for i, event in enumerate(events):
            event_type = event.get('event_type', 'Unknown')
            
            # Get color for event type
            if event_type in event_colors:
                r, g, b = event_colors[event_type]
                pdf.set_fill_color(r, g, b)
                pdf.set_text_color(255, 255, 255)  # White text for colored backgrounds
            else:
                pdf.set_fill_color(200, 200, 200)  # Gray for unknown events
                pdf.set_text_color(51, 51, 51)     # Dark text
            
            # Event header
            timestamp = event.get('timestamp', '')
            pdf.cell(0, 8, txt=f"* {event_type} - {timestamp}", ln=True, fill=True)
            
            # Event details
            pdf.set_fill_color(255, 255, 255)  # White background for details
            pdf.set_text_color(51, 51, 51)     # Dark text
            
            details_text = ""
            if 'details' in event:
                details_text += f"Details: {event['details']}"
            if 'ear_value' in event:
                if details_text:
                    details_text += " | "
                details_text += f"EAR: {event['ear_value']}"
            
            if details_text:
                pdf.cell(0, 6, txt=details_text, ln=True, fill=True)
            
            pdf.ln(2)  # Small spacing between events
    
    # Summary section
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_fill_color(76, 175, 80)  # Green background
    pdf.set_text_color(255, 255, 255)  # White text
    pdf.cell(0, 10, txt="Trip Summary", ln=True, fill=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 11)
    pdf.set_fill_color(240, 248, 255)  # Light blue background
    pdf.set_text_color(51, 51, 51)     # Dark text
    
    total_events = len(events)
    event_counts = {}
    for event in events:
        event_type = event.get('event_type', 'Unknown')
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    pdf.cell(0, 8, txt=f"Total Events: {total_events}", ln=True, fill=True)
    
    for event_type, count in event_counts.items():
        pdf.cell(0, 8, txt=f"- {event_type}: {count} occurrence(s)", ln=True, fill=True)
    
    # Footer
    pdf.ln(10)
    pdf.set_font("Arial", '', 8)
    pdf.set_text_color(128, 128, 128)  # Gray text
    pdf.cell(0, 5, txt=f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.cell(0, 5, txt="Real-Time Driver Monitoring System", ln=True, align='C')
    
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

# --- BEAUTIFUL LOGIN PAGE ---
def show_login():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Main background with animated gradient */
        body { 
            background: linear-gradient(-45deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            font-family: 'Poppins', sans-serif;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Floating particles effect */
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            z-index: -1;
        }
        
        .particle {
            position: absolute;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            animation: float 6s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.7; }
            50% { transform: translateY(-20px) rotate(180deg); opacity: 1; }
        }
        
        /* Main container for split layout */
        .main-container {
            display: flex;
            min-height: 100vh;
            align-items: center;
            justify-content: space-between;
            padding: 0 2rem;
        }
        
        /* Left side - Info card */
        .info-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1), 0 0 0 1px rgba(255, 255, 255, 0.2);
            padding: 3rem 2.5rem 2.5rem 2.5rem;
            width: 400px;
            margin-right: 2rem;
            font-family: 'Poppins', sans-serif;
            border: 1px solid rgba(255, 255, 255, 0.3);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
        }
        
        .info-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }
        
        .info-card:hover::before {
            left: 100%;
        }
        
        .info-card:hover {
            box-shadow: 0 30px 80px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(255, 255, 255, 0.3);
            transform: translateY(-5px) scale(1.02);
        }
        
        /* Right side - Login card */
        
        
        .login-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }
        
        .login-card:hover::before {
            left: 100%;
        }
        
        .login-card:hover {
            box-shadow: 0 30px 80px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(255, 255, 255, 0.3);
            transform: translateY(-5px) scale(1.02);
        }
        
        /* Enhanced logo */
        .login-logo {
            display: block;
            margin: 0 auto 1.5rem auto;
            width: 90px;
            height: 90px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
        }
        
        .login-logo:hover {
            transform: rotate(360deg) scale(1.1);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        }
        
        /* Enhanced title */
        .login-title {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 0.3em;
            letter-spacing: 1px;
            font-family: 'Poppins', sans-serif;
            text-shadow: 0 2px 10px rgba(102, 126, 234, 0.1);
        }
        
        /* Enhanced divider */
        .login-divider {
            width: 80px;
            height: 5px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            border-radius: 3px;
            margin: 1em auto 1.5em auto;
            position: relative;
            overflow: hidden;
        }
        
        .login-divider::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.6), transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        /* Enhanced subtitle */
        .login-sub {
            text-align: center;
            color: #6c7a89;
            font-size: 1.1rem;
            font-weight: 400;
            margin-bottom: 2em;
            letter-spacing: 0.5px;
            line-height: 1.6;
        }
        
        /* Enhanced form styling */
        .stTextInput > div > div > input {
            border-radius: 15px !important;
            border: 2px solid #e1e8ed !important;
            padding: 12px 20px !important;
            font-size: 16px !important;
            transition: all 0.3s ease !important;
            background: rgba(255, 255, 255, 0.9) !important;
            color: black !important;
        }
        .stTextInput > div > div > input::placeholder {
            color: black !important;
            opacity: 0.8 !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
            transform: translateY(-2px) !important;
        }
        
        /* Enhanced button styling */
        .stButton > button {
            border-radius: 15px !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            padding: 12px 30px !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            color: white !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4) !important;
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
        }
        
        /* Enhanced text styling */
        .register-text {
            text-align: center;
            color: #6c7a89;
            font-size: 14px;
            margin: 1.5rem 0 1rem 0;
            font-weight: 400;
        }
        
        /* Hide Streamlit default elements */
        .main > div:first-child {
            padding-top: 0 !important;
        }
        
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        </style>
        
        <!-- Floating particles -->
        <div class="particles">
            <div class="particle" style="left: 10%; top: 20%; width: 20px; height: 20px; animation-delay: 0s;"></div>
            <div class="particle" style="left: 80%; top: 40%; width: 15px; height: 15px; animation-delay: 1s;"></div>
            <div class="particle" style="left: 20%; top: 70%; width: 25px; height: 25px; animation-delay: 2s;"></div>
            <div class="particle" style="left: 70%; top: 10%; width: 18px; height: 18px; animation-delay: 3s;"></div>
            <div class="particle" style="left: 90%; top: 80%; width: 22px; height: 22px; animation-delay: 4s;"></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("""
    <div class="login-card">
        <h1 style="text-align: center; color: #667eea; font-size: 1.8rem; margin-bottom: 1.5rem; font-weight: 600;">🚘 Real-Time Driver Monitoring System</h1>
    """, unsafe_allow_html=True)
    
    username = st.text_input('👤 Username', key='login_username', placeholder="Enter your username")
    password = st.text_input('🔒 Password', type='password', key='login_password', placeholder="Enter your password")
    login_btn = st.button('🔓 Sign In', key='login_btn')
    
    st.markdown('<div class="register-text">Don\'t have an account?</div>', unsafe_allow_html=True)
    register_btn = st.button('📝 Create Account', key='login_register_btn')
    st.markdown("</div>", unsafe_allow_html=True)
    
    return username, password, login_btn, register_btn

# --- BEAUTIFUL REGISTER PAGE ---
def show_register():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Main background with animated gradient */
        body { 
            background: linear-gradient(-45deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            font-family: 'Poppins', sans-serif;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Floating particles effect */
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            z-index: -1;
        }
        
        .particle {
            position: absolute;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            animation: float 6s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.7; }
            50% { transform: translateY(-20px) rotate(180deg); opacity: 1; }
        }
        
        /* Enhanced register card */
        .login-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1), 0 0 0 1px rgba(255, 255, 255, 0.2);
            padding: 3rem 2.5rem 2.5rem 2.5rem;
            max-width: 450px;
            margin: 50px auto 0 auto;
            font-family: 'Poppins', sans-serif;
            border: 1px solid rgba(255, 255, 255, 0.3);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
        }
        
        .login-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }
        
        .login-card:hover::before {
            left: 100%;
        }
        
        .login-card:hover {
            box-shadow: 0 30px 80px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(255, 255, 255, 0.3);
            transform: translateY(-5px) scale(1.02);
        }
        
        /* Enhanced logo */
        .login-logo {
            display: block;
            margin: 0 auto 1.5rem auto;
            width: 90px;
            height: 90px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
        }
        
        .login-logo:hover {
            transform: rotate(360deg) scale(1.1);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        }
        
        /* Enhanced title */
        .login-title {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 0.3em;
            letter-spacing: 1px;
            font-family: 'Poppins', sans-serif;
            text-shadow: 0 2px 10px rgba(102, 126, 234, 0.1);
        }
        
        /* Enhanced divider */
        .login-divider {
            width: 80px;
            height: 5px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            border-radius: 3px;
            margin: 1em auto 1.5em auto;
            position: relative;
            overflow: hidden;
        }
        
        .login-divider::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.6), transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        /* Enhanced subtitle */
        .login-sub {
            text-align: center;
            color: #e1e8ed;
            font-size: 1.1rem;
            font-weight: 400;
            margin-bottom: 2em;
            letter-spacing: 0.5px;
            line-height: 1.6;
        }
        
        /* Enhanced form styling */
        .stTextInput > div > div > input {
            border-radius: 15px !important;
            border: 2px solid #e1e8ed !important;
            padding: 12px 20px !important;
            font-size: 16px !important;
            transition: all 0.3s ease !important;
            color: black !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
            transform: translateY(-2px) !important;
        }
        
        /* Enhanced selectbox styling */
        .stSelectbox > div > div > div {
            border-radius: 15px !important;
            border: 2px solid #e1e8ed !important;
        }
        
        .stSelectbox > div > div > div:focus-within {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        
        /* Enhanced button styling */
        .stButton > button {
            border-radius: 15px !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            padding: 12px 30px !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            color: white !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4) !important;
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
        }
        
        /* Back button specific styling */
        .stButton > button:last-child {
            background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%) !important;
            box-shadow: 0 8px 25px rgba(149, 165, 166, 0.3) !important;
        }
        
        .stButton > button:last-child:hover {
            background: linear-gradient(135deg, #7f8c8d 0%, #6c7b7c 100%) !important;
            box-shadow: 0 12px 35px rgba(149, 165, 166, 0.4) !important;
        }
        
        /* Hide Streamlit default elements */
        .main > div:first-child {
            padding-top: 0 !important;
        }
        
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        </style>
        
        <!-- Floating particles -->
        <div class="particles">
            <div class="particle" style="left: 10%; top: 20%; width: 20px; height: 20px; animation-delay: 0s;"></div>
            <div class="particle" style="left: 80%; top: 40%; width: 15px; height: 15px; animation-delay: 1s;"></div>
            <div class="particle" style="left: 20%; top: 70%; width: 25px; height: 25px; animation-delay: 2s;"></div>
            <div class="particle" style="left: 70%; top: 10%; width: 18px; height: 18px; animation-delay: 3s;"></div>
            <div class="particle" style="left: 90%; top: 80%; width: 22px; height: 22px; animation-delay: 4s;"></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    
    
    new_username = st.text_input('👤 Choose a username', key='register_username', placeholder="Enter your desired username")
    new_password = st.text_input('🔒 Choose a password', type='password', key='register_password', placeholder="Enter a strong password")
    role = st.selectbox('🧑‍💼 Select your role', ['driver', 'manager'], key='register_role')
    register_btn = st.button('✅ Create Account', key='register_btn')
    
    st.markdown("<div style='text-align:center; margin-top: 1.5rem;'>", unsafe_allow_html=True)
    back_btn = st.button('⬅️ Back to Login', key='register_back_btn')
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    return new_username, new_password, role, register_btn, back_btn

# --- LOGIN SYSTEM ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None

if not st.session_state.logged_in:
    if st.session_state.current_page == 'home':
        username, password, login_btn, register_btn = show_login()
        if login_btn:
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
        if register_btn:
            go_to('register')
        st.stop()
    elif st.session_state.current_page == 'register':
        new_username, new_password, role, register_btn, back_btn = show_register()
        if register_btn:
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
        if back_btn:
            go_back()
        st.stop()

# --- MAIN APP NAVIGATION ---
if st.session_state.current_page == 'dashboard':
    if st.session_state.role == 'driver':
        # Enhanced Driver Dashboard with beautiful styling
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Driver Dashboard styling */
        .driver-main-header {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            padding: 2.5rem;
            border-radius: 20px;
            margin: 0 0 2rem 0;
            box-shadow: 0 15px 35px rgba(59, 130, 246, 0.3);
            color: white;
            text-align: center;
            width: 100%;
        }
        
        .driver-main-title {
            font-family: 'Poppins', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }
        
        .driver-main-subtitle {
            font-family: 'Poppins', sans-serif;
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 300;
        }
        
        .trip-card {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(20px);
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .trip-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
        }
        
        .trip-header {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .alert-card {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 8px 25px rgba(239, 68, 68, 0.3);
            transition: all 0.3s ease;
            text-align: center;
        }
        
        .alert-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(239, 68, 68, 0.4);
        }
        
        .safe-card {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
            transition: all 0.3s ease;
            text-align: center;
        }
        
        .safe-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(16, 185, 129, 0.4);
        }
        
        .section-header {
            font-family: 'Poppins', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: #3b82f6;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #3b82f6;
            width: 100%;
        }
        
        .back-button {
            background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .back-button:hover {
            background: linear-gradient(135deg, #4b5563 0%, #374151 100%);
            transform: translateY(-2px);
        }
        
        /* Enhanced button styling */
        .stButton > button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Download button styling */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
            color: white !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
        }
        
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #d97706 0%, #b45309 100%) !important;
            transform: translateY(-2px) !important;
        }
        
        /* Full width layout */
        .main .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        /* Remove side margins */
        .stApp > div:first-child {
            padding: 0 !important;
        }
        
        /* Ensure full width for all containers */
        .stApp {
            max-width: 100% !important;
        }
        
        /* Remove default Streamlit padding */
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Driver Dashboard Header
        st.markdown(f"""
        <div class="driver-main-header">
            <div class="driver-main-title">Real-Time Driver Monitoring System</div>
            <div class="driver-main-subtitle">Welcome back, {st.session_state.username}!</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Back button with styling
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            if st.button('⬅️ Back to Login', key='driver_main_back_btn'):
                st.session_state.logged_in = False
                st.session_state.role = None
                st.session_state.username = None
                st.session_state.nav_stack = ['home']
                st.session_state.current_page = 'home'
                st.rerun()

        # --- SIDEBAR NAVIGATION FOR DRIVER ---
        with st.sidebar:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 15px;
                margin-bottom: 1rem;
                text-align: center;
            ">
                <h3 style="margin: 0; font-family: 'Poppins', sans-serif;">🚘 Driver Menu</h3>
            </div>
            """, unsafe_allow_html=True)
            
            driver_option = st.radio(
                "Select an option:",
                ["Start Monitoring", "Download Report"],
                key="driver_sidebar_option"
            )

        if driver_option == "Start Monitoring":
            # Trip selection UI
            st.markdown('<div class="section-header">🚗 Trip Details</div>', unsafe_allow_html=True)
            
            start_point = st.text_input("🚀 Start Point", placeholder="Enter your starting location")
            destination = st.text_input("🎯 Destination", placeholder="Enter your destination")
            trip_started = st.session_state.get('trip_started', False)
            
            if not trip_started:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Start Monitoring", key="start_monitoring_btn", use_container_width=True):
                        if not start_point or not destination:
                            st.warning("⚠️ Please enter both start point and destination.")
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
                            st.success(f"✅ Trip started from {start_point} to {destination}!")
                            st.rerun()
            else:
                # Fetch current trip details
                trips = get_trips_for_driver(st.session_state.username)
                current_trip = None
                for t in trips:
                    if str(t['_id']) == st.session_state.current_trip_id:
                        current_trip = t
                        break
                
                st.markdown(f"""
                <div class="trip-card">
                    <div class="trip-header">
                        <h3 style="margin: 0; font-size: 1.3rem;">🔄 Trip in Progress</h3>
                    </div>
                    <div style="text-align: center; color: #3b82f6; font-size: 1.1rem; font-weight: 600;">
                        {current_trip['start_point']} → {current_trip['destination']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # --- ALERT FUNCTIONS (KEEPING ALL FUNCTIONALITY INTACT) ---
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
                
                # Enhanced Alert Display
                st.markdown('<div class="section-header">📊 Real-Time Monitoring</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                drowsiness_alert = col1.empty()
                yawn_alert = col2.empty()
                phone_alert = col3.empty()
                
                
                
                run = st.checkbox('🎥 Start Camera', key='camera_checkbox')
                
                # KEEPING ALL CAMERA FUNCTIONALITY INTACT
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
                                    if ear < 0.20:
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
                                    # Yawn detection with debug
                                    if st.session_state.get('debug_yawn', False):
                                        is_yawn, mouth_ratio, mouth_distance, face_width = is_yawning(landmarks, debug=True)
                                        st.sidebar.write(f"Yawn debug: ratio={mouth_ratio:.3f}, dist={mouth_distance:.1f}, width={face_width:.1f}")
                                        if is_yawn:
                                            yawning_detected = True
                                            cv2.putText(frame, "YAWNING", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                                            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                            log_ride({
                                                'timestamp': current_time,
                                                'event_type': 'Yawning',
                                                'details': f'Mouth ratio: {mouth_ratio:.3f}, dist: {mouth_distance:.1f}, width: {face_width:.1f}',
                                                'driver': st.session_state.username,
                                                'trip_id': st.session_state.current_trip_id
                                            })
                                    else:
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
                            
                            # Enhanced Alert Display
                            if drowsiness_detected:
                                drowsiness_alert.markdown("""
                                <div class="alert-card">
                                    <h3 style="margin: 0;">😴 Drowsiness Detected</h3>
                                    <p style="margin: 0.5rem 0; opacity: 0.9;">Please stay alert!</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                drowsiness_alert.markdown("""
                                <div class="safe-card">
                                    <h3 style="margin: 0;">✅ Alert</h3>
                                    <p style="margin: 0.5rem 0; opacity: 0.9;">Stay focused!</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            if yawning_detected:
                                yawn_alert.markdown("""
                                <div class="alert-card">
                                    <h3 style="margin: 0;">🥱 Yawning Detected</h3>
                                    <p style="margin: 0.5rem 0; opacity: 0.9;">Take a break if needed!</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                yawn_alert.markdown("""
                                <div class="safe-card">
                                    <h3 style="margin: 0;">✅ Alert</h3>
                                    <p style="margin: 0.5rem 0; opacity: 0.9;">Stay focused!</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            if phone_detected:
                                phone_alert.markdown("""
                                <div class="alert-card">
                                    <h3 style="margin: 0;">📱 Phone Detected</h3>
                                    <p style="margin: 0.5rem 0; opacity: 0.9;">Focus on driving!</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                phone_alert.markdown("""
                                <div class="safe-card">
                                    <h3 style="margin: 0;">✅ Alert</h3>
                                    <p style="margin: 0.5rem 0; opacity: 0.9;">Stay focused!</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            stframe.image(frame, channels="BGR")
                    cap.release()
                
                # End Trip Button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button('🏁 End Trip', key='end_trip_btn', use_container_width=True):
                        # Mark trip as ended (KEEPING FUNCTIONALITY INTACT)
                        from pymongo import MongoClient
                        client = MongoClient("mongodb://localhost:27017/IDP")
                        db = client["IDP"]
                        db["trips"].update_one({'_id': ObjectId(st.session_state.current_trip_id)}, {"$set": {"end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}})
                        st.session_state.trip_started = False
                        st.session_state.current_trip_id = None
                        st.success("✅ Trip ended successfully!")
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
                    st.markdown('<div class="section-header">📋 Trip Summary</div>', unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="trip-card">
                        <div class="trip-header">
                            <h3 style="margin: 0; font-size: 1.3rem;">✅ Trip Completed</h3>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0;">
                            <div style="background: #f8fafc; padding: 1rem; border-radius: 10px; border-left: 4px solid #3b82f6; color: #1f2937;">
                                <strong style="color: #3b82f6;">🚀 Start Point:</strong><br>{trip['start_point']}
                            </div>
                            <div style="background: #f8fafc; padding: 1rem; border-radius: 10px; border-left: 4px solid #3b82f6; color: #1f2937;">
                                <strong style="color: #3b82f6;">🎯 Destination:</strong><br>{trip['destination']}
                            </div>
                            <div style="background: #f8fafc; padding: 1rem; border-radius: 10px; border-left: 4px solid #3b82f6; color: #1f2937;">
                                <strong style="color: #3b82f6;">⏰ Start Time:</strong><br>{trip['start_time']}
                            </div>
                            <div style="background: #f8fafc; padding: 1rem; border-radius: 10px; border-left: 4px solid #3b82f6; color: #1f2937;">
                                <strong style="color: #3b82f6;">🏁 End Time:</strong><br>{trip.get('end_time', 'N/A')}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Get events for this trip
                    all_events = get_rides_for_driver(st.session_state.username)
                    trip_events = [e for e in all_events if e.get('trip_id') == st.session_state.current_trip_id]
                    pdf_bytes = generate_trip_pdf(trip, trip_events)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label="📄 Download Trip Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"trip_report_{trip['start_point']}_to_{trip['destination']}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
        
        elif driver_option == "Download Report":
            st.markdown('<div class="section-header">📥 Download Report</div>', unsafe_allow_html=True)
            
            trips = get_trips_for_driver(st.session_state.username)
            if not trips:
                st.markdown("""
                <div class="trip-card" style="text-align: center;">
                    <h3 style="color: #6b7280; margin: 0;">📋 No Trips Recorded</h3>
                    <p style="color: #3b82f6; margin: 0.5rem 0;">You haven't recorded any trips yet. Start monitoring to generate reports.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for trip in trips:
                    st.markdown(f"""
                    <div class="trip-card">
                        <div class="trip-header">
                            <h3 style="margin: 0; font-size: 1.3rem;">Trip: {trip['start_point']} → {trip['destination']}</h3>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0;">
                            <div style="background: #f8fafc; padding: 1rem; border-radius: 10px; border-left: 4px solid #3b82f6; color: #1f2937;">
                                <strong style="color: #3b82f6;">⏰ Start Time:</strong><br>{trip['start_time']}
                            </div>
                            <div style="background: #f8fafc; padding: 1rem; border-radius: 10px; border-left: 4px solid #3b82f6; color: #1f2937;">
                                <strong style="color: #3b82f6;">🏁 End Time:</strong><br>{trip.get('end_time', 'N/A')}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    all_events = get_rides_for_driver(st.session_state.username)
                    trip_events = [e for e in all_events if e.get('trip_id') == str(trip['_id'])]
                    pdf_bytes = generate_trip_pdf(trip, trip_events)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label=f"📄 Download Trip Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"trip_report_{trip['start_point']}_to_{trip['destination']}.pdf",
                            mime="application/pdf",
                            key=f"driver_download_pdf_{trip['_id']}",
                            use_container_width=True
                        )
    elif st.session_state.role == 'manager':
        # Enhanced Fleet Manager Dashboard with beautiful styling
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Dashboard styling */
        .dashboard-header {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            padding: 2.5rem;
            border-radius: 20px;
            margin: 0 0 2rem 0;
            box-shadow: 0 15px 35px rgba(79, 70, 229, 0.3);
            color: white;
            text-align: center;
            width: 100%;
        }
        
        .dashboard-title {
            font-family: 'Poppins', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }
        
        .dashboard-subtitle {
            font-family: 'Poppins', sans-serif;
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 300;
        }
        
        .stats-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            backdrop-filter: blur(20px);
            border-radius: 15px;
            padding: 2rem;
            margin: 0.5rem;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
            height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            width: 100%;
            color: white;
        }
        
        .stats-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
        }
        
        .driver-card {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem;
            box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
            transition: all 0.3s ease;
            height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            width: 100%;
        }
        
        .driver-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(59, 130, 246, 0.4);
        }
        
        .section-header {
            font-family: 'Poppins', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: #4f46e5;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #4f46e5;
            width: 100%;
        }
        
        .back-button {
            background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .back-button:hover {
            background: linear-gradient(135deg, #7f8c8d 0%, #6c7b7c 100%);
            transform: translateY(-2px);
        }
        
        /* Enhanced button styling */
        .stButton > button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Radio button styling */
        .stRadio > div > label {
            font-weight: 500 !important;
            color: #4f46e5 !important;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 10px !important;
            overflow: hidden !important;
        }
        
        /* Download button styling */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #76c893 0%, #52b69a 100%) !important;
            color: white !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
        }
        
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #52b69a 0%, #34a0a4 100%) !important;
            transform: translateY(-2px) !important;
        }
        
        /* Full width layout */
        .main .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        /* Remove side margins */
        .stApp > div:first-child {
            padding: 0 !important;
        }
        
        /* Ensure full width for all containers */
        .stApp {
            max-width: 100% !important;
        }
        
        /* Remove default Streamlit padding */
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Dashboard Header
        st.markdown(f"""
        <div class="dashboard-header">
            <div class="dashboard-title">Fleet Manager Dashboard</div>
            <div class="dashboard-subtitle">Welcome back, {st.session_state.username}!</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Back button with styling
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            if st.button('⬅️ Back to Login', key='manager_back_btn'):
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
        
        # Statistics Cards
        st.markdown('<div class="section-header">📊 Fleet Statistics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="stats-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <h3 style="color: white; margin: 0; font-size: 1.2rem;">Total Drivers</h3>
                <p style="font-size: 2.5rem; font-weight: 700; color: white; margin: 0.5rem 0;">{len(all_drivers)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stats-card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                <h3 style="color: white; margin: 0; font-size: 1.2rem;">My Drivers</h3>
                <p style="font-size: 2.5rem; font-weight: 700; color: white; margin: 0.5rem 0;">{len(my_drivers)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stats-card" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">
                <h3 style="color: white; margin: 0; font-size: 1.2rem;">Unassigned</h3>
                <p style="font-size: 2.5rem; font-weight: 700; color: white; margin: 0.5rem 0;">{len(unassigned_drivers)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            all_rides = get_all_rides()
            st.markdown(f"""
            <div class="stats-card" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);">
                <h3 style="color: white; margin: 0; font-size: 1.2rem;">Total Events</h3>
                <p style="font-size: 2.5rem; font-weight: 700; color: white; margin: 0.5rem 0;">{len(all_rides)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Driver Management Section
        st.markdown('<div class="section-header">👥 Driver Management</div>', unsafe_allow_html=True)
        
        section = st.radio(
            'Select an action:',
            ['Show & Assign Unassigned Drivers', 'Show My Drivers'],
            key='manager_section',
            horizontal=True
        )
        
        if section == 'Show & Assign Unassigned Drivers':
            st.markdown('<h4 style="color: #4f46e5; margin: 1rem 0;">Unassigned Drivers</h4>', unsafe_allow_html=True)
            
            if unassigned_drivers:
                st.markdown(f"""
                <div class="driver-card" style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);">
                    <h4 style="margin: 0 0 1rem 0;">Available Drivers: {len(unassigned_drivers)}</h4>
                    <p style="margin: 0; opacity: 0.9;">{', '.join(unassigned_drivers)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                selected_driver = st.selectbox(
                    'Select a driver to assign to yourself:',
                    unassigned_drivers,
                    key='assign_driver_select'
                )
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button('🚗 Assign Selected Driver', key='assign_selected_driver_btn', use_container_width=True):
                        if selected_driver:
                            assign_driver_to_manager(selected_driver, manager_username)
                            st.success(f"✅ Driver '{selected_driver}' successfully assigned to you!")
                            st.rerun()
                        else:
                            st.warning('⚠️ Please select a driver to assign.')
            else:
                st.markdown("""
                <div class="stats-card" style="text-align: center; background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                    <h3 style="color: white; margin: 0;">🎉 All Drivers Assigned!</h3>
                    <p style="color: white; margin: 0.5rem 0; opacity: 0.9;">Great job! All drivers are now assigned to fleet managers.</p>
                </div>
                """, unsafe_allow_html=True)
        
        elif section == 'Show My Drivers':
            st.markdown('<h4 style="color: #4f46e5; margin: 1rem 0;">Drivers Under Your Management</h4>', unsafe_allow_html=True)
            
            if my_drivers:
                for i, drv in enumerate(my_drivers):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"""
                        <div class="driver-card">
                            <h4 style="margin: 0 0 0.5rem 0;">🚗 {drv}</h4>
                            <p style="margin: 0; opacity: 0.9;">Driver ID: {i+1}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        if st.button(f"📊 View Dashboard", key=f"view_{drv}", use_container_width=True):
                            go_to(f'driver_dashboard_{drv}')
                            st.rerun()
            else:
                st.markdown("""
                <div class="stats-card" style="text-align: center; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">
                    <h3 style="color: white; margin: 0;">📋 No Drivers Assigned</h3>
                    <p style="color: white; margin: 0.5rem 0; opacity: 0.9;">You don't have any drivers assigned to you yet. Assign drivers from the unassigned list.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Event Logs Section
        st.markdown('<div class="section-header">📈 Driver Event Logs</div>', unsafe_allow_html=True)
        
        if all_rides:
            # Enhanced dataframe display with better styling
            df = pd.DataFrame(all_rides)
            
            # Clean and format the dataframe
            if not df.empty:
                # Format timestamp column if it exists
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Add event type icons and color coding
                def format_event_type(event_type):
                    event_icons = {
                        'Drowsiness': '😴',
                        'Yawning': '🥱',
                        'Phone Usage': '📱',
                        'Lane Change': '🛣️',
                        'Speed': '⚡'
                    }
                    return f"{event_icons.get(event_type, '⚠️')} {event_type}"
                
                if 'event_type' in df.columns:
                    df['Event Type'] = df['event_type'].apply(format_event_type)
                    df = df.drop('event_type', axis=1)
                
                # Reorder columns for better readability
                column_order = ['timestamp', 'Event Type', 'driver', 'details', 'ear_value', 'trip_id']
                existing_columns = [col for col in column_order if col in df.columns]
                df = df[existing_columns + [col for col in df.columns if col not in existing_columns]]
                
                # Rename columns for better display
                df = df.rename(columns={
                    'timestamp': '📅 Timestamp',
                    'driver': '👤 Driver',
                    'details': '📝 Details',
                    'ear_value': '👁️ EAR Value',
                    'trip_id': '🚗 Trip ID'
                })
            
            # Enhanced table styling
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                border-radius: 20px;
                padding: 2rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                border: 1px solid rgba(255,255,255,0.2);
                margin: 1rem 0;
            ">
                <h3 style="
                    color: #4f46e5;
                    font-family: 'Poppins', sans-serif;
                    font-size: 1.3rem;
                    font-weight: 600;
                    margin: 0 0 1.5rem 0;
                    text-align: center;
                ">📊 Event Summary</h3>
            """, unsafe_allow_html=True)
            
            # Show summary statistics
            if not df.empty:
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                
                with col1:
                    st.metric(
                        label="Total Events",
                        value=len(df),
                        delta=None
                    )
                
                with col2:
                    if 'Event Type' in df.columns:
                        drowsiness_count = len(df[df['Event Type'].str.contains('😴', na=False)])
                        st.metric(
                            label="Drowsiness Events",
                            value=drowsiness_count,
                            delta=None
                        )
                
                with col3:
                    if 'Event Type' in df.columns:
                        phone_count = len(df[df['Event Type'].str.contains('📱', na=False)])
                        st.metric(
                            label="Phone Usage",
                            value=phone_count,
                            delta=None
                        )
                
                with col4:
                    if 'Event Type' in df.columns:
                        yawn_count = len(df[df['Event Type'].str.contains('🥱', na=False)])
                        st.metric(
                            label="Yawning Events",
                            value=yawn_count,
                            delta=None
                        )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Enhanced dataframe with better styling
            st.markdown("""
            <div style="
                background: white;
                border-radius: 15px;
                padding: 1.5rem;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                border: 1px solid rgba(255,255,255,0.2);
                margin: 1rem 0;
            ">
                <h4 style="
                    color: #4f46e5;
                    font-family: 'Poppins', sans-serif;
                    font-size: 1.1rem;
                    font-weight: 600;
                    margin: 0 0 1rem 0;
                ">📋 Detailed Event Log</h4>
            """, unsafe_allow_html=True)
            
            # Display the dataframe with enhanced styling
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Enhanced download section
            csv = df.to_csv(index=False)
            
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                border-radius: 15px;
                padding: 1.5rem;
                box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
                margin: 1rem 0;
                text-align: center;
            ">
                <h4 style="
                    color: white;
                    font-family: 'Poppins', sans-serif;
                    font-size: 1.1rem;
                    font-weight: 600;
                    margin: 0 0 1rem 0;
                ">📥 Export Data</h4>
            """, unsafe_allow_html=True)
            
            with col2:
                st.download_button(
                    label="📊 Download CSV Report",
                    data=csv,
                    file_name=f"fleet_event_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="stats-card" style="text-align: center; background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);">
                <h3 style="color: white; margin: 0;">📊 No Events Recorded</h3>
                <p style="color: white; margin: 0.5rem 0; opacity: 0.9;">No driver events have been recorded yet. Events will appear here once drivers start their monitoring sessions.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.stop()
# --- DRIVER DASHBOARD FOR MANAGER ---
if st.session_state.current_page.startswith('driver_dashboard_'):
    driver_username = st.session_state.current_page.replace('driver_dashboard_', '')
    
    # Enhanced Driver Dashboard with beautiful styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Driver Dashboard styling */
    .driver-dashboard-header {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 0 0 2rem 0;
        box-shadow: 0 15px 35px rgba(59, 130, 246, 0.3);
        color: white;
        text-align: center;
        width: 100%;
    }
    
    .driver-dashboard-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .driver-dashboard-subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .trip-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .trip-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    .trip-header {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .trip-details {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .trip-detail-item {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
    }
    
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #3b82f6;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3b82f6;
        width: 100%;
    }
    
    .back-button {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .back-button:hover {
        background: linear-gradient(135deg, #4b5563 0%, #374151 100%);
        transform: translateY(-2px);
    }
    
    /* Enhanced button styling */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Full width layout */
    .main .block-container {
        max-width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Remove side margins */
    .stApp > div:first-child {
        padding: 0 !important;
    }
    
    /* Ensure full width for all containers */
    .stApp {
        max-width: 100% !important;
    }
    
    /* Remove default Streamlit padding */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Driver Dashboard Header
    st.markdown(f"""
    <div class="driver-dashboard-header">
        <div class="driver-dashboard-title">Driver Dashboard</div>
        <div class="driver-dashboard-subtitle">Monitoring {driver_username}'s Performance</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Back button with styling
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button('⬅️ Back to Fleet Manager', key='driver_back_btn'):
            go_back()
            st.rerun()
    
    # Get driver data
    trips = get_trips_for_driver(driver_username)
    all_events = get_rides_for_driver(driver_username)
    
    # Driver Statistics
    st.markdown('<div class="section-header">📊 Driver Statistics</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="trip-card" style="text-align: center;">
            <h3 style="color: #3b82f6; margin: 0; font-size: 1.2rem;">Total Trips</h3>
            <p style="font-size: 2.5rem; font-weight: 700; color: #3b82f6; margin: 0.5rem 0;">{len(trips)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="trip-card" style="text-align: center;">
            <h3 style="color: #10b981; margin: 0; font-size: 1.2rem;">Total Events</h3>
            <p style="font-size: 2.5rem; font-weight: 700; color: #10b981; margin: 0.5rem 0;">{len(all_events)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        completed_trips = len([t for t in trips if 'end_time' in t])
        st.markdown(f"""
        <div class="trip-card" style="text-align: center;">
            <h3 style="color: #f59e0b; margin: 0; font-size: 1.2rem;">Completed Trips</h3>
            <p style="font-size: 2.5rem; font-weight: 700; color: #f59e0b; margin: 0.5rem 0;">{completed_trips}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Trip Details Section
    st.markdown('<div class="section-header">🚗 Trip Details</div>', unsafe_allow_html=True)
    
    if not trips:
        st.markdown("""
        <div class="trip-card" style="text-align: center;">
            <h3 style="color: #6b7280; margin: 0;">📋 No Trips Recorded</h3>
            <p style="color: #3b82f6; margin: 0.5rem 0;">This driver hasn't recorded any trips yet.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, trip in enumerate(trips):
            trip_events = [e for e in all_events if e.get('trip_id') == str(trip['_id'])]
            st.markdown(f"""
**Trip #{i+1}: {trip['start_point']} → {trip['destination']}**
- 🚀 **Start Point:** {trip['start_point']}
- 🎯 **Destination:** {trip['destination']}
- ⏰ **Start Time:** {trip['start_time']}
- 🏁 **End Time:** {trip.get('end_time', 'Ongoing')}
- 📊 **Events:** {len(trip_events)} events recorded
- 📈 **Status:** {'✅ Completed' if 'end_time' in trip else '🔄 Active'}
            """)
            if trip_events:
                pdf_bytes = generate_trip_pdf(trip, trip_events)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label=f"📄 Download Trip Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"trip_report_{trip['start_point']}_to_{trip['destination']}.pdf",
                        mime="application/pdf",
                        key=f"download_pdf_{trip['_id']}",
                        use_container_width=True
                    )
    
    st.stop()