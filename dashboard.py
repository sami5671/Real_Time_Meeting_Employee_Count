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
import matplotlib.pyplot as plt

# ------------------------------
# Firebase Init
# ------------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase/serviceAccountKey.json")
    initialize_app(cred)

db = firestore.client()

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Smart Attendance",
    page_icon="🎓",
    layout="wide",
)

# ------------------------------
# Custom SaaS Styling
# ------------------------------
st.markdown(
    """
<style>
.main {
    background-color: #0f172a;
}
.stMetric {
    background: #111827;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #1f2937;
    color: white !important;  /* <-- make text white */
}
.stMetric span[data-testid="stMetricValue"] {
    color: white !important;  /* <-- specifically for metric values */
}
.stMetric div[data-testid="stMetricDelta"] {
    color: white !important;  /* <-- for delta text if used */
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🎓 Smart Face Attendance System")

# ------------------------------
# Load Encodings
# ------------------------------
data = pickle.loads(open("data/encodings.pickle", "rb").read())
DATASET_PATH = "data/dataset"

students_list = []
if os.path.exists(DATASET_PATH):
    students_list = os.listdir(DATASET_PATH)

# ------------------------------
# Session State
# ------------------------------
if "running" not in st.session_state:
    st.session_state.running = False

if "attendance" not in st.session_state:
    st.session_state.attendance = {}

if "cap" not in st.session_state:
    st.session_state.cap = None

if "session_id" not in st.session_state:
    st.session_state.session_id = ""

if "view_history" not in st.session_state:
    st.session_state.view_history = False

# ------------------------------
# Sidebar (Meeting Setup)
# ------------------------------
st.sidebar.title("⚙️ Meeting Control")

course_code = st.sidebar.text_input("Course Code", "CSE401")
duration = st.sidebar.number_input("Meeting Duration (minutes)", min_value=1, value=10)
late_minutes = st.sidebar.number_input("Late Entry Allowed (minutes)", min_value=0, value=5)

selected_students = st.sidebar.multiselect("Select Students", students_list)

start = st.sidebar.button("🚀 Start Meeting")
stop = st.sidebar.button("🛑 Stop Meeting")
history_btn = st.sidebar.button("📊 Previous Meetings")

if history_btn:
    st.session_state.view_history = True
    st.session_state.running = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None

# ------------------------------
# Start Meeting
# ------------------------------
if start:
    st.session_state.running = True
    st.session_state.view_history = False

    st.session_state.start_time = datetime.now()
    st.session_state.end_time = datetime.now() + timedelta(minutes=duration)
    st.session_state.late_deadline = datetime.now() + timedelta(minutes=late_minutes)
    st.session_state.session_id = datetime.now().strftime("%Y-%m-%d") + "_" + course_code

    attendance = {}
    for s in selected_students:
        sid, name = s.split("_", 1)
        attendance[sid] = {"name": name, "status": "Absent", "time": ""}
    st.session_state.attendance = attendance
    st.session_state.cap = cv2.VideoCapture(0)

# ------------------------------
# Stop Meeting
# ------------------------------
if stop and st.session_state.running:
    st.session_state.running = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None

    session_id = st.session_state.session_id
    attendance = st.session_state.attendance
    csv_file = f"attendance_{session_id}.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Student ID", "Name", "Status", "Time"])
        for sid, info in attendance.items():
            writer.writerow([sid, info["name"], info["status"], info["time"]])

    st.success(f"CSV Saved: {csv_file}")

# ------------------------------
# Attendance Function
# ------------------------------
def mark_attendance(student_info):
    sid, name = student_info.split("_", 1)
    attendance = st.session_state.attendance
    if sid not in attendance or attendance[sid]["status"] != "Absent":
        return

    now = datetime.now()
    status = "Present" if now <= st.session_state.late_deadline else "Late"
    attendance[sid]["status"] = status
    attendance[sid]["time"] = now.strftime("%H:%M:%S")

    # Save to Firebase
    db.collection("attendance").document(st.session_state.session_id).collection(
        "students"
    ).document(sid).set({
        "name": name,
        "student_id": sid,
        "status": status,
        "timestamp": now,
    })

# ------------------------------
# KPI Cards
# ------------------------------
if st.session_state.attendance and not st.session_state.view_history:
    df = pd.DataFrame.from_dict(st.session_state.attendance, orient="index")
    total = len(df)
    present = len(df[df["status"] == "Present"])
    late = len(df[df["status"] == "Late"])
    absent = total - present - late

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👨‍🎓 Total Students", total)
    c2.metric("✅ Present", present)
    c3.metric("⏰ Late", late)
    c4.metric("❌ Absent", absent)

# ------------------------------
# Previous Meetings Functions (CSV + Firebase)
# ------------------------------
CSV_PATH = "."

def load_csv_sessions():
    files = [f for f in os.listdir(CSV_PATH) if f.startswith("attendance_") and f.endswith(".csv")]
    return files

def load_csv_attendance(file_name):
    df = pd.read_csv(os.path.join(CSV_PATH, file_name))
    return df

def plot_attendance_charts(df):
    status_counts = df["Status"].value_counts()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🥧 Attendance Distribution")
        fig1, ax1 = plt.subplots()
        ax1.pie(status_counts, labels=status_counts.index, autopct="%1.0f%%", startangle=90)
        st.pyplot(fig1)

    with c2:
        st.subheader("📊 Attendance Bar Chart")
        st.bar_chart(status_counts)

    st.subheader("📈 Attendance Trend")
    trend_df = df.groupby("Status").size().reset_index(name="Count")
    st.line_chart(trend_df.set_index("Status"))

# ------------------------------
# Main Layout
# ------------------------------
if st.session_state.view_history:
    # Only show previous meeting analytics
    st.subheader("📊 Previous Meeting Analytics")
    csv_sessions = load_csv_sessions()
    firebase_sessions = [s.id for s in db.collection("attendance").stream()]
    all_sessions = sorted(list(set(csv_sessions + firebase_sessions)))

    if all_sessions:
        selected_session = st.selectbox("Select Meeting Session", all_sessions)

        if selected_session.endswith(".csv"):
            df = load_csv_attendance(selected_session)
        else:
            # Firebase session
            students = db.collection("attendance").document(selected_session).collection("students").stream()
            data_list = []
            for s in students:
                d = s.to_dict()
                data_list.append({
                    "Student ID": d["student_id"],
                    "Name": d["name"],
                    "Status": d["status"],
                    "Time": d["timestamp"]
                })
            df = pd.DataFrame(data_list)

        if not df.empty:
            st.dataframe(df, use_container_width=True)
            plot_attendance_charts(df)
    else:
        st.info("No previous sessions found.")

else:
    # Show live camera + attendance table only when not viewing history
    left, right = st.columns([2, 1])
    frame_window = None

    with left:
        st.subheader("📷 Live Camera Feed")
        frame_window = st.empty()

    with right:
        st.subheader("📊 Attendance Table")
        if st.session_state.attendance:
            df = pd.DataFrame.from_dict(st.session_state.attendance, orient="index")
            st.dataframe(df, use_container_width=True)

    # ------------------------------
    # Face Recognition Loop
    # ------------------------------
    if st.session_state.running:
        cap = st.session_state.cap
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, boxes)

            for encoding, box in zip(encodings, boxes):
                matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.45)
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
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            frame_window.image(frame, channels="BGR")
        time.sleep(0.1)
        st.rerun()