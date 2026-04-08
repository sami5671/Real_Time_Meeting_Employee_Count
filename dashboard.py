import os
import pickle
import time
from datetime import datetime, timedelta

import cv2
import face_recognition
import firebase_admin
import pandas as pd
import requests
import streamlit as st
from firebase_admin import credentials, firestore, initialize_app

# ------------------------------
# Telegram Config
# ------------------------------
TOKEN = "8600372885:AAH9MaxJx1xYhcZfRWDKFsbHKRUkOAfJaM8"


def send_telegram_message(chat_id, message):
    """Send a Telegram message. Returns True on success, False on failure."""
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        res = requests.post(
            url,
            data={"chat_id": str(chat_id), "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        result = res.json()
        if result.get("ok"):
            print(f"✅ Telegram sent to {chat_id}")
            return True
        else:
            print(f"❌ Telegram failed for {chat_id}: {result}")
            return False
    except Exception as e:
        print(f"❌ Telegram Exception for {chat_id}: {e}")
        return False


# ------------------------------
# Firebase Init
# ------------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase/serviceAccountKey.json")
    initialize_app(cred)

db = firestore.client()

# ------------------------------
# Streamlit Setup
# ------------------------------
st.set_page_config(page_title="Smart Attendance", layout="wide")
st.title("🎓 Smart Face Attendance System")


# ------------------------------
# Load Encodings
# ------------------------------
@st.cache_resource
def load_encodings():
    with open("data/encodings.pickle", "rb") as f:
        return pickle.load(f)


data = load_encodings()

DATASET_PATH = "data/dataset"
students_list = os.listdir(DATASET_PATH) if os.path.exists(DATASET_PATH) else []

# ------------------------------
# Session State Defaults
# ------------------------------
defaults = {
    "running": False,
    "attendance": {},
    "cap": None,
    "start_time": None,
    "late_deadline": None,
    "session_id": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("⚙️ Control Panel")
course_code = st.sidebar.text_input("Course Code", "CSE401")
late_minutes = st.sidebar.number_input("Late Allow (min)", 0, 60, 2)
selected_students = st.sidebar.multiselect("Select Students", students_list)

start = st.sidebar.button("🚀 Start Session")
stop = st.sidebar.button("🛑 Stop Session")

# ------------------------------
# Attendance Marker
# ------------------------------
def mark_attendance(student_folder):
    """Mark a detected student as Present or Late."""
    parts = student_folder.split("_")
    if len(parts) < 3:
        return

    sid = parts[0]
    name = parts[1]
    # chat_id is the last part (handles names with underscores)
    chat_id = parts[-1]

    if sid not in st.session_state.attendance:
        return

    record = st.session_state.attendance[sid]

    # Only update if still Absent
    if record["status"] == "Absent":
        now = datetime.now()
        status = "Present" if now <= st.session_state.late_deadline else "Late"

        record["status"] = status
        record["time"] = now.strftime("%H:%M:%S")

        # Firebase update
        try:
            db.collection("attendance").document(st.session_state.session_id)\
                .collection("students").document(sid).set({
                    "name": name,
                    "status": status,
                    "timestamp": now,
                })
        except Exception as e:
            print(f"Firebase error for {sid}: {e}")

        # If they arrived late, notify immediately
        if status == "Late" and not record["notified"]:
            message = (
                f"⚠️ *Late Entry Detected*\n\n"
                f"👤 Name: {record['name']}\n"
                f"🆔 ID: {sid}\n"
                f"📌 Status: Late\n"
                f"🕐 Time: {record['time']}\n"
                f"📚 Session: {st.session_state.session_id}\n\n"
                f"Please join immediately."
            )
            send_telegram_message(record["chat_id"], message)
            record["notified"] = True


# ------------------------------
# Late Detection & Notification
# ------------------------------
def check_and_notify_late_absent():
    """
    After the deadline, mark remaining Absent students as Late
    and send Telegram notifications ONCE per student.
    """
    now = datetime.now()
    if st.session_state.late_deadline is None:
        return
    if now <= st.session_state.late_deadline:
        return  # Deadline not passed yet

    for sid, info in st.session_state.attendance.items():
        # Only act on students who are still Absent and not yet notified
        if info["status"] == "Absent" and not info["notified"]:
            info["status"] = "Late"
            info["time"] = now.strftime("%H:%M:%S")
            info["notified"] = True  # Set BEFORE sending to prevent double-send

            message = (
                f"🚨 *Absence Alert*\n\n"
                f"👤 Name: {info['name']}\n"
                f"🆔 ID: {sid}\n"
                f"📌 Status: Absent / Late\n"
                f"🕐 Deadline Passed: {st.session_state.late_deadline.strftime('%H:%M:%S')}\n"
                f"📚 Session: {st.session_state.session_id}\n\n"
                f"You have been marked Late. Please join immediately."
            )

            success = send_telegram_message(info["chat_id"], message)

            # Firebase update
            try:
                db.collection("attendance").document(st.session_state.session_id)\
                    .collection("students").document(sid).set({
                        "name": info["name"],
                        "status": "Late",
                        "timestamp": now,
                    })
            except Exception as e:
                print(f"Firebase error for {sid}: {e}")

            st.toast(
                f"{'✅' if success else '❌'} Telegram {'sent' if success else 'FAILED'} → {info['name']}",
                icon="📩",
            )


# ------------------------------
# Start Session
# ------------------------------
if start:
    if not selected_students:
        st.sidebar.error("Please select at least one student.")
    else:
        st.session_state.running = True
        st.session_state.start_time = datetime.now()
        st.session_state.late_deadline = datetime.now() + timedelta(minutes=late_minutes)
        st.session_state.session_id = (
            f"{course_code}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        )

        temp = {}
        for s in selected_students:
            parts = s.split("_")
            if len(parts) >= 3:
                sid = parts[0]
                name = parts[1]
                chat_id = parts[-1]  # last part = chat_id (safe for names with _)
                temp[sid] = {
                    "name": name,
                    "chat_id": chat_id,
                    "status": "Absent",
                    "time": "--",
                    "notified": False,
                }

        st.session_state.attendance = temp

        # Release any old capture
        if st.session_state.cap is not None:
            st.session_state.cap.release()
        st.session_state.cap = cv2.VideoCapture(0)

        st.sidebar.success(f"Session started: {st.session_state.session_id}")

# ------------------------------
# Stop Session
# ------------------------------
if stop:
    st.session_state.running = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.sidebar.info("Session stopped.")

# ------------------------------
# UI Placeholders
# ------------------------------
frame_window = st.empty()
status_bar = st.empty()
table_window = st.empty()

# ------------------------------
# MAIN LOOP
# ------------------------------
if st.session_state.running:
    cap = st.session_state.cap

    if cap is None or not cap.isOpened():
        st.error("❌ Camera not accessible. Please restart the session.")
        st.session_state.running = False
    else:
        ret, frame = cap.read()

        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)

            for encoding, box in zip(encodings, boxes):
                matches = face_recognition.compare_faces(
                    data["encodings"], encoding, tolerance=0.45
                )
                matched_name = "Unknown"

                if True in matches:
                    matched_idxs = [i for i, b in enumerate(matches) if b]
                    counts = {}
                    for i in matched_idxs:
                        n = data["names"][i]
                        counts[n] = counts.get(n, 0) + 1
                    matched_name = max(counts, key=counts.get)
                    mark_attendance(matched_name)

                top, r, bottom, l = box
                color = (0, 255, 0) if matched_name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (l, top), (r, bottom), color, 2)
                cv2.putText(
                    frame,
                    matched_name,
                    (l, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            frame_window.image(frame, channels="BGR")
        else:
            st.warning("⚠️ Could not read frame from camera.")

        # ------------------------------
        # Check late / absent notifications
        # ------------------------------
        check_and_notify_late_absent()

        # ------------------------------
        # Status bar
        # ------------------------------
        now = datetime.now()
        deadline = st.session_state.late_deadline
        if deadline and now <= deadline:
            remaining = int((deadline - now).total_seconds())
            status_bar.info(
                f"⏱️ Session: `{st.session_state.session_id}` | "
                f"Late deadline in: **{remaining}s**"
            )
        else:
            status_bar.warning(
                f"⏰ Session: `{st.session_state.session_id}` | "
                f"Late deadline **passed** at {deadline.strftime('%H:%M:%S') if deadline else '--'}"
            )

        # ------------------------------
        # Attendance Table
        # ------------------------------
        if st.session_state.attendance:
            df = pd.DataFrame.from_dict(
                st.session_state.attendance, orient="index"
            )[["name", "status", "time", "notified"]]
            df.index.name = "Student ID"

            def color_status(val):
                if val == "Present":
                    return "background-color: #d4edda; color: #155724"
                elif val == "Late":
                    return "background-color: #fff3cd; color: #856404"
                elif val == "Absent":
                    return "background-color: #f8d7da; color: #721c24"
                return ""

            styled = df.style.applymap(color_status, subset=["status"])
            table_window.dataframe(styled, use_container_width=True)

        time.sleep(0.5)
        st.rerun()

elif not st.session_state.running and st.session_state.attendance:
    # Show final table after session stops
    st.subheader("📋 Final Attendance Report")
    df = pd.DataFrame.from_dict(st.session_state.attendance, orient="index")[
        ["name", "status", "time"]
    ]
    df.index.name = "Student ID"
    st.dataframe(df, use_container_width=True)

    present = sum(1 for v in st.session_state.attendance.values() if v["status"] == "Present")
    late = sum(1 for v in st.session_state.attendance.values() if v["status"] == "Late")
    absent = sum(1 for v in st.session_state.attendance.values() if v["status"] == "Absent")
    total = len(st.session_state.attendance)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", total)
    c2.metric("✅ Present", present)
    c3.metric("⚠️ Late", late)
    c4.metric("❌ Absent", absent)