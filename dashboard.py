import csv
import os
import pickle
import time
from datetime import datetime, timedelta

import cv2
import face_recognition
import firebase_admin
import pandas as pd
import streamlit as st
from firebase_admin import credentials, firestore, initialize_app

# =========================
# Firebase Initialization
# =========================
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase/serviceAccountKey.json")
    initialize_app(cred)

db = firestore.client()

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Smart Attendance System", layout="wide")
st.title("🎓 Smart Attendance System Dashboard")

# =========================
# Load Encodings
# =========================
data = pickle.loads(open("data/encodings.pickle", "rb").read())

# =========================
# Load Students
# =========================
DATASET_PATH = "data/dataset"
students_list = []

if os.path.exists(DATASET_PATH):
    students_list = os.listdir(DATASET_PATH)

# =========================
# Session State
# =========================
if "running" not in st.session_state:
    st.session_state.running = False

if "attendance" not in st.session_state:
    st.session_state.attendance = {}

if "session_id" not in st.session_state:
    st.session_state.session_id = ""

if "cap" not in st.session_state:
    st.session_state.cap = None


# =========================
# Meeting Setup
# =========================
st.header("📅 Create Meeting Session")

col1, col2 = st.columns(2)

with col1:
    course_code = st.text_input("Course Code", "CSE401")

with col2:
    duration = st.number_input("Meeting Duration (minutes)", min_value=1, value=10)

late_minutes = st.number_input("Late Entry Allowed (minutes)", min_value=0, value=5)

selected_students = st.multiselect("Select Students For Meeting", students_list)

# =========================
# Buttons
# =========================
col1, col2 = st.columns(2)

start = col1.button("🚀 Start Meeting")
stop = col2.button("🛑 Stop Meeting")

# =========================
# Start Meeting
# =========================
if start:

    if len(selected_students) == 0:
        st.warning("Select students first")
        st.stop()

    st.session_state.running = True
    st.session_state.start_time = datetime.now()
    st.session_state.end_time = datetime.now() + timedelta(minutes=duration)
    st.session_state.late_deadline = datetime.now() + timedelta(minutes=late_minutes)

    st.session_state.session_id = (
        datetime.now().strftime("%Y-%m-%d") + "_" + course_code
    )

    attendance = {}

    for s in selected_students:
        sid, name = s.split("_")
        attendance[sid] = {"name": name, "status": "Absent", "time": ""}

    st.session_state.attendance = attendance

    st.session_state.cap = cv2.VideoCapture(0)

# =========================
# Stop Meeting
# =========================
if stop and st.session_state.running:

    st.session_state.running = False

    if st.session_state.cap:
        st.session_state.cap.release()

    st.success("Meeting Stopped")

    session_id = st.session_state.session_id
    attendance = st.session_state.attendance

    csv_file = f"attendance_{session_id}.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Student ID", "Name", "Status", "Time"])

        for sid, info in attendance.items():
            writer.writerow([sid, info["name"], info["status"], info["time"]])

    st.success(f"CSV Saved: {csv_file}")


# =========================
# Mark Attendance
# =========================
def mark_attendance(student_info):

    sid, name = student_info.split("_")

    attendance = st.session_state.attendance

    if sid not in attendance:
        return

    if attendance[sid]["status"] != "Absent":
        return

    now = datetime.now()

    if now <= st.session_state.late_deadline:
        status = "Present"
    else:
        status = "Late"

    attendance[sid]["status"] = status
    attendance[sid]["time"] = now.strftime("%H:%M:%S")

    db.collection("attendance").document(st.session_state.session_id).collection(
        "students"
    ).document(sid).set(
        {
            "name": name,
            "student_id": sid,
            "status": status,
            "timestamp": now,
        }
    )


# =========================
# Camera Loop
# =========================
frame_window = st.empty()

if st.session_state.running:

    cap = st.session_state.cap

    ret, frame = cap.read()

    if ret:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding, box in zip(encodings, boxes):

            matches = face_recognition.compare_faces(
                data["encodings"], encoding, tolerance=0.45
            )

            name = "Unknown"

            if True in matches:

                matched_idxs = [i for (i, b) in enumerate(matches) if b]

                counts = {}

                for i in matched_idxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                name = max(counts, key=counts.get)

                mark_attendance(name)

            top, right, bottom, left = box

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.putText(
                frame,
                name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        frame_window.image(frame, channels="BGR")

    time.sleep(0.05)
    st.rerun()

# =========================
# Real Time Dashboard
# =========================
st.divider()
st.header("📊 Live Attendance")

if st.session_state.attendance:

    df = pd.DataFrame.from_dict(st.session_state.attendance, orient="index")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Students", len(df))

    with col2:
        st.metric("Present", len(df[df["status"] == "Present"]))

    with col3:
        st.metric("Late", len(df[df["status"] == "Late"]))

    st.dataframe(df, use_container_width=True)
