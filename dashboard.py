import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from collections import Counter

import cv2
import face_recognition
import firebase_admin
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
from firebase_admin import credentials, firestore, initialize_app

# ──────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS · Smart Attendance",
    layout="wide",
    page_icon="🛰️",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL CSS INJECTION — Futuristic Dark Theme
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

    /* ── Root & Background ── */
    :root {
        --cyan:      #00e5ff;
        --cyan-dim:  #00b4cc;
        --cyan-glow: rgba(0,229,255,0.18);
        --green:     #00ff9d;
        --yellow:    #ffd600;
        --red:       #ff3d5a;
        --bg-base:   #040b14;
        --bg-card:   #071525;
        --bg-card2:  #0a1e30;
        --border:    rgba(0,229,255,0.18);
        --text:      #e0f7ff;
        --text-dim:  #7ab8cc;
        --font-head: 'Orbitron', monospace;
        --font-body: 'Rajdhani', sans-serif;
        --font-mono: 'Share Tech Mono', monospace;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg-base) !important;
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,229,255,0.07) 0%, transparent 70%),
            repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(0,229,255,0.03) 40px),
            repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,229,255,0.03) 40px);
        color: var(--text) !important;
        font-family: var(--font-body) !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050f1c 0%, #061320 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        border: 1px solid var(--cyan-dim) !important;
        color: var(--cyan) !important;
        font-family: var(--font-head) !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.1em !important;
        border-radius: 4px !important;
        transition: all 0.2s !important;
        width: 100% !important;
        margin-bottom: 6px !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--cyan-glow) !important;
        box-shadow: 0 0 12px var(--cyan-glow) !important;
    }

    /* Main Area Buttons (like Refresh Data) */
    .stButton > button {
        color: #000000 !important;
    }
    .stButton > button:hover {
        color: #ffffff !important;
    }

    /* ── Main title ── */
    h1 { display: none !important; }  /* hide default, we use custom */

    /* ── Inputs & Selects ── */
    .stSelectbox select {
        background: var(--bg-card2) !important;
        border: 1px solid var(--border) !important;
        color: var(--cyan) !important;
        font-family: var(--font-mono) !important;
        border-radius: 4px !important;
    }
    .stTextInput input, .stNumberInput input, .stMultiSelect [data-baseweb="select"] {
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        color: #000000 !important;
        font-family: var(--font-mono) !important;
        border-radius: 4px !important;
    }
    
    .stMultiSelect svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* Number Input Buttons Style */
    .stNumberInput button {
        background: #ff3d5a !important;
        border-radius: 4px !important;
        border: none !important;
        transition: background 0.2s !important;
    }
    .stNumberInput button:hover {
        background: #cc3048 !important; /* dimmed red */
    }
    .stNumberInput button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    .stMultiSelect span { color: var(--text) !important; }
    
    /* Multiselect Placeholder color */
    .stMultiSelect [data-baseweb="select"] div[data-testid="stMarkdownContainer"] p,
    .stMultiSelect [data-baseweb="select"] div {
        color: #000000 !important;
    }
    
    /* Multiselect icons (dropdown arrow and cross) */
    .stMultiSelect svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    label, .stTextInput label, .stNumberInput label,
    .stSelectbox label, .stMultiSelect label {
        color: var(--text-dim) !important;
        font-family: var(--font-body) !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    /* ── DataFrames ── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    .dvn-scroller { background: var(--bg-card) !important; }

    /* ── Alerts / Info / Warning ── */
    .stAlert { border-radius: 6px !important; font-family: var(--font-mono) !important; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 16px 20px !important;
        position: relative !important;
    }
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, var(--cyan), transparent);
    }
    [data-testid="stMetricValue"] {
        color: var(--cyan) !important;
        font-family: var(--font-head) !important;
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-dim) !important;
        font-family: var(--font-body) !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        font-size: 0.7rem !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] p {
        color: var(--cyan) !important;
        font-family: var(--font-head) !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.1em !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: var(--cyan-dim); border-radius: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# CUSTOM HEADER
# ──────────────────────────────────────────────
st.markdown(
    """
    <div style="
        display:flex; align-items:center; gap:20px;
        padding:18px 0 10px; border-bottom:1px solid rgba(0,229,255,0.15);
        margin-bottom:24px;">
        <div style="font-size:2rem;">🛰️</div>
        <div>
            <div style="font-family:'Orbitron',monospace; font-size:1.4rem;
                color:#00e5ff; letter-spacing:0.18em; font-weight:700;
                text-shadow:0 0 18px rgba(0,229,255,0.5);">
                NEXUS ATTENDANCE
            </div>
            <div style="font-family:'Rajdhani',sans-serif; font-size:0.75rem;
                color:#7ab8cc; letter-spacing:0.25em; text-transform:uppercase;">
                Smart Face Recognition · Real-time Analytics
            </div>
        </div>
        <div style="margin-left:auto; font-family:'Share Tech Mono',monospace;
            font-size:0.7rem; color:#00e5ff; opacity:0.6; text-align:right;">
            SYS ONLINE<br>
            <span id="clock" style="color:#00ff9d;">──────────</span>
        </div>
    </div>
    <script>
    if (window.clockInterval) clearInterval(window.clockInterval);
    function tick(){
        const now=new Date();
        const t=now.toTimeString().slice(0,8);
        const el=document.getElementById('clock');
        if(el) el.textContent=t;
    }
    window.clockInterval = setInterval(tick,1000); tick();
    </script>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# HELPER — styled card wrapper
# ──────────────────────────────────────────────
def card(title: str, icon: str = ""):
    """Return HTML for a card header (use inside st.markdown)."""
    return f"""
    <div style="
        background:linear-gradient(135deg,#071525 0%,#0a1e30 100%);
        border:1px solid rgba(0,229,255,0.18);
        border-radius:10px; padding:16px 20px; margin-bottom:12px;
        box-shadow:0 0 24px rgba(0,229,255,0.05);
        position:relative; overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:1px;
            background:linear-gradient(90deg,#00e5ff,transparent);"></div>
        <div style="font-family:'Orbitron',monospace; font-size:0.75rem;
            color:#00e5ff; letter-spacing:0.18em; text-transform:uppercase;
            margin-bottom:2px;">{icon} {title}</div>
    </div>"""

# ──────────────────────────────────────────────
# PLOTLY THEME DEFAULTS
# ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Rajdhani", color="#e0f7ff", size=12),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,229,255,0.2)"),
    xaxis=dict(gridcolor="rgba(0,229,255,0.06)", zerolinecolor="rgba(0,229,255,0.1)"),
    yaxis=dict(gridcolor="rgba(0,229,255,0.06)", zerolinecolor="rgba(0,229,255,0.1)"),
)

# ──────────────────────────────────────────────
# TELEGRAM CONFIG
# ──────────────────────────────────────────────
TOKEN = "8600372885:AAH9MaxJx1xYhcZfRWDKFsbHKRUkOAfJaM8"


def send_telegram_message(chat_id, message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        res = requests.post(
            url,
            data={"chat_id": str(chat_id), "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        result = res.json()
        return (chat_id, result.get("ok", False))
    except Exception as e:
        print(f"❌ Telegram Exception for {chat_id}: {e}")
        return (chat_id, False)


def send_telegram_to_many(targets: list):
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(send_telegram_message, t["chat_id"], t["message"]): t
            for t in targets
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                t = futures[future]
                results.append((t["chat_id"], False))
    return results


# ──────────────────────────────────────────────
# FIREBASE INIT
# ──────────────────────────────────────────────
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase/serviceAccountKey.json")
    initialize_app(cred)

db = firestore.client()

# ──────────────────────────────────────────────
# LOAD ENCODINGS
# ──────────────────────────────────────────────
@st.cache_resource
def load_encodings():
    with open("data/encodings.pickle", "rb") as f:
        return pickle.load(f)

data = load_encodings()

DATASET_PATH = "data/dataset"
students_list = os.listdir(DATASET_PATH) if os.path.exists(DATASET_PATH) else []

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
defaults = {
    "running": False,
    "attendance": {},
    "cap": None,
    "start_time": None,
    "late_deadline": None,
    "session_id": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """<div style="font-family:'Orbitron',monospace; font-size:0.65rem;
            color:#00e5ff; letter-spacing:0.2em; text-transform:uppercase;
            border-bottom:1px solid rgba(0,229,255,0.15); padding-bottom:10px;
            margin-bottom:16px;">⚙ Control Panel</div>""",
        unsafe_allow_html=True,
    )
    course_code = st.text_input("Course Code", "CSE401")
    late_minutes = st.number_input("Late Allowance (min)", 0, 60, 2)
    selected_students = st.multiselect("Enrolled Students", students_list)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    start = st.button("🚀  START SESSION")
    stop  = st.button("🛑  STOP SESSION")

    st.markdown(
        """<div style="margin-top:24px; padding:12px;
            background:rgba(0,229,255,0.04); border:1px solid rgba(0,229,255,0.1);
            border-radius:6px; font-family:'Share Tech Mono',monospace;
            font-size:0.65rem; color:#7ab8cc; line-height:1.8;">
            SYS · NEXUS v2.0<br>
            ENGINE · HOG / DLIB<br>
            STORE  · FIRESTORE<br>
            NOTIFY · TELEGRAM
        </div>""",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# FIREBASE ANALYTICS LOADER
# ──────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_firebase_analytics():
    """
    Pulls all sessions + student records from Firestore.
    Returns a flat DataFrame with columns:
        session_id, course_code, student_id, name, status, timestamp
    """
    rows = []
    try:
        sessions = db.collection("attendance").stream()
        for sess in sessions:
            sess_data = sess.to_dict()
            sid = sess.id
            course = sess_data.get("course_code", "—")
            students = (
                db.collection("attendance").document(sid).collection("students").stream()
            )
            for stu in students:
                d = stu.to_dict()
                ts = d.get("timestamp")
                rows.append(
                    {
                        "session_id": sid,
                        "course_code": course,
                        "student_id": stu.id,
                        "name": d.get("name", "—"),
                        "status": d.get("status", "—"),
                        "timestamp": ts.replace(tzinfo=None) if ts else None,
                    }
                )
    except Exception as e:
        st.sidebar.warning(f"Firebase read error: {e}")
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        columns=["session_id", "course_code", "student_id", "name", "status", "timestamp"]
    )


# ──────────────────────────────────────────────
# ATTENDANCE CORE FUNCTIONS
# ──────────────────────────────────────────────
def mark_attendance(student_folder):
    parts = student_folder.split("_")
    if len(parts) < 3:
        return
    sid, name, chat_id = parts[0], parts[1], parts[-1]
    if sid not in st.session_state.attendance:
        return
    record = st.session_state.attendance[sid]
    if record["status"] != "Absent":
        return
    now    = datetime.now()
    status = "Present" if now <= st.session_state.late_deadline else "Late"
    record["status"] = status
    record["time"]   = now.strftime("%H:%M:%S")
    try:
        db.collection("attendance").document(st.session_state.session_id).collection(
            "students"
        ).document(sid).set({"name": name, "status": status, "timestamp": now})
    except Exception as e:
        print(f"Firebase error for {sid}: {e}")
    if status == "Late" and not record["notified"]:
        record["notified"] = True
        msg = (
            f"⚠️ *Late Entry Detected*\n\n"
            f"👤 Name: {record['name']}\n🆔 ID: {sid}\n"
            f"📌 Status: Late\n🕐 Time: {record['time']}\n"
            f"📚 Session: {st.session_state.session_id}\n\nPlease join immediately."
        )
        send_telegram_message(record["chat_id"], msg)


def check_and_notify_late_absent():
    now = datetime.now()
    if st.session_state.late_deadline is None or now <= st.session_state.late_deadline:
        return
    targets = []
    for sid, info in st.session_state.attendance.items():
        if info["status"] == "Absent" and not info["notified"]:
            info.update({"status": "Late", "time": now.strftime("%H:%M:%S"), "notified": True})
            msg = (
                f"🚨 *Late / Absent Alert*\n\n"
                f"👤 Name: {info['name']}\n🆔 ID: {sid}\n📌 Status: Late\n"
                f"🕐 Deadline: {st.session_state.late_deadline.strftime('%H:%M:%S')}\n"
                f"📚 Session: {st.session_state.session_id}\n\nYou have been marked Late."
            )
            targets.append({"sid": sid, "chat_id": info["chat_id"], "name": info["name"], "message": msg})
            try:
                db.collection("attendance").document(st.session_state.session_id).collection(
                    "students"
                ).document(sid).set({"name": info["name"], "status": "Late", "timestamp": now})
            except Exception as e:
                print(f"Firebase error for {sid}: {e}")
    if not targets:
        return
    results = send_telegram_to_many(targets)
    id_to_name = {t["chat_id"]: t["name"] for t in targets}
    for chat_id, success in results:
        name = id_to_name.get(str(chat_id), str(chat_id))
        st.toast(f"{'✅' if success else '❌'} {'Sent' if success else 'FAILED'} → {name}", icon="📩")


# ──────────────────────────────────────────────
# START / STOP SESSION
# ──────────────────────────────────────────────
if start:
    if not selected_students:
        st.sidebar.error("Select at least one student.")
    else:
        st.session_state.running       = True
        st.session_state.start_time    = datetime.now()
        st.session_state.late_deadline = datetime.now() + timedelta(minutes=late_minutes)
        st.session_state.session_id    = f"{course_code}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        try:
            db.collection("attendance").document(st.session_state.session_id).set(
                {
                    "session_id":   st.session_state.session_id,
                    "course_code":  course_code,
                    "start_time":   st.session_state.start_time,
                    "late_deadline": st.session_state.late_deadline,
                    "created_at":   firestore.SERVER_TIMESTAMP,
                }
            )
        except Exception as e:
            print(f"Firebase session write error: {e}")

        temp = {}
        for s in selected_students:
            parts = s.split("_")
            if len(parts) >= 3:
                sid = parts[0]
                temp[sid] = {
                    "name": parts[1], "chat_id": parts[-1],
                    "status": "Absent", "time": "--", "notified": False,
                }
        st.session_state.attendance = temp
        
        # ── INITIALIZE ALL AS ABSENT IN FIREBASE ──
        for sid, info in temp.items():
            try:
                db.collection("attendance").document(st.session_state.session_id).collection("students").document(sid).set({
                    "name": info["name"],
                    "status": "Absent",
                    "timestamp": st.session_state.start_time
                })
            except Exception as e:
                print(f"Firebase initial record error for {sid}: {e}")

        if st.session_state.cap is not None:
            st.session_state.cap.release()
        st.session_state.cap = cv2.VideoCapture(0)
        st.sidebar.success(f"Session launched: {st.session_state.session_id}")

if stop:
    st.session_state.running = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.sidebar.info("Session terminated.")

# ──────────────────────────────────────────────
# TAB LAYOUT
# ──────────────────────────────────────────────
tab_live, tab_analytics = st.tabs(["🎥  LIVE SESSION", "📊  ANALYTICS"])

# ══════════════════════════════════════════════
# TAB 1 — LIVE SESSION
# ══════════════════════════════════════════════
with tab_live:
    frame_window  = st.empty()
    status_bar    = st.empty()
    table_window  = st.empty()

    if st.session_state.running:
        cap = st.session_state.cap

        if cap is None or not cap.isOpened():
            st.error("❌ Camera not accessible. Restart session.")
            st.session_state.running = False
        else:
            ret, frame = cap.read()
            if ret:
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes    = face_recognition.face_locations(rgb, model="hog")
                encodings= face_recognition.face_encodings(rgb, boxes)

                for encoding, box in zip(encodings, boxes):
                    matches      = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.45)
                    matched_name = "Unknown"
                    if True in matches:
                        idxs   = [i for i, b in enumerate(matches) if b]
                        counts = {}
                        for i in idxs:
                            n = data["names"][i]
                            counts[n] = counts.get(n, 0) + 1
                        matched_name = max(counts, key=counts.get)
                        mark_attendance(matched_name)

                    top, r, bottom, l = box
                    color = (0, 229, 255) if matched_name != "Unknown" else (255, 61, 90)
                    cv2.rectangle(frame, (l, top), (r, bottom), color, 2)
                    # glow effect via double rectangle
                    cv2.rectangle(frame, (l-1, top-1), (r+1, bottom+1), (color[0]//3, color[1]//3, color[2]//3), 1)
                    cv2.putText(frame, matched_name, (l, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                frame_window.image(frame, channels="BGR", use_container_width=True)
            else:
                st.warning("⚠️ Camera frame unreadable.")

            check_and_notify_late_absent()

            # Status bar
            now      = datetime.now()
            deadline = st.session_state.late_deadline
            if deadline and now <= deadline:
                remaining = int((deadline - now).total_seconds())
                status_bar.markdown(
                    f"""<div style="background:rgba(0,229,255,0.06);border:1px solid rgba(0,229,255,0.2);
                        border-radius:6px;padding:10px 16px;font-family:'Share Tech Mono',monospace;
                        font-size:0.75rem;color:#00e5ff;display:flex;gap:24px;">
                        <span>📡 SESSION · <b>{st.session_state.session_id}</b></span>
                        <span>⏱ DEADLINE IN · <b style="color:#00ff9d;">{remaining}s</b></span>
                        <span>👥 ENROLLED · <b>{len(st.session_state.attendance)}</b></span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                status_bar.markdown(
                    f"""<div style="background:rgba(255,214,0,0.05);border:1px solid rgba(255,214,0,0.25);
                        border-radius:6px;padding:10px 16px;font-family:'Share Tech Mono',monospace;
                        font-size:0.75rem;color:#ffd600;">
                        ⚠ SESSION · {st.session_state.session_id} &nbsp;|&nbsp;
                        DEADLINE PASSED · {deadline.strftime('%H:%M:%S') if deadline else '--'}
                    </div>""",
                    unsafe_allow_html=True,
                )

            # Live attendance table
            if st.session_state.attendance:
                df = pd.DataFrame.from_dict(
                    st.session_state.attendance, orient="index"
                )[["name", "status", "time", "notified"]]
                df.index.name = "Student ID"

                STATUS_COLOR = {
                    "Present": "background-color:#003d20;color:#00ff9d",
                    "Late":    "background-color:#3d2c00;color:#ffd600",
                    "Absent":  "background-color:#3d0010;color:#ff3d5a",
                }

                def color_status(val):
                    return STATUS_COLOR.get(val, "")

                styled = df.style.applymap(color_status, subset=["status"])
                table_window.dataframe(styled, use_container_width=True)

            time.sleep(0.5)
            st.rerun()

    elif not st.session_state.running and st.session_state.attendance:
        # Final report
        st.markdown(
            """<div style="font-family:'Orbitron',monospace;font-size:0.75rem;
                color:#00e5ff;letter-spacing:0.18em;margin-bottom:16px;">
                📋 FINAL SESSION REPORT</div>""",
            unsafe_allow_html=True,
        )
        df = pd.DataFrame.from_dict(st.session_state.attendance, orient="index")[
            ["name", "status", "time"]
        ]
        df.index.name = "Student ID"

        present = sum(1 for v in st.session_state.attendance.values() if v["status"] == "Present")
        late    = sum(1 for v in st.session_state.attendance.values() if v["status"] == "Late")
        absent  = sum(1 for v in st.session_state.attendance.values() if v["status"] == "Absent")
        total   = len(st.session_state.attendance)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Enrolled", total)
        c2.metric("✅ Present",      present)
        c3.metric("⚠️ Late",         late)
        c4.metric("❌ Absent",        absent)

        st.dataframe(df, use_container_width=True)

        # Mini donut for quick view
        fig = go.Figure(
            go.Pie(
                labels=["Present", "Late", "Absent"],
                values=[present, late, absent],
                hole=0.65,
                marker=dict(colors=["#00ff9d", "#ffd600", "#ff3d5a"],
                            line=dict(color="#040b14", width=3)),
                textfont=dict(family="Rajdhani", color="#e0f7ff"),
            )
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Session Breakdown", font=dict(family="Orbitron", color="#00e5ff", size=13)),
            annotations=[dict(text=f"{total}<br>TOTAL", x=0.5, y=0.5,
                              font=dict(family="Orbitron", size=14, color="#00e5ff"),
                              showarrow=False)],
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Idle state
        st.markdown(
            """<div style="text-align:center;padding:80px 0;">
                <div style="font-size:4rem;margin-bottom:16px;">🛰️</div>
                <div style="font-family:'Orbitron',monospace;font-size:1rem;
                    color:#00e5ff;letter-spacing:0.2em;opacity:0.7;">
                    AWAITING SESSION INITIALIZATION
                </div>
                <div style="font-family:'Rajdhani',sans-serif;font-size:0.85rem;
                    color:#7ab8cc;margin-top:8px;">
                    Configure course settings in the Control Panel and press START SESSION
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════
# TAB 2 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════
with tab_analytics:
    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄  Refresh Data"):
            st.cache_data.clear()

    df_all = load_firebase_analytics()

    if df_all.empty:
        st.info("No historical data found in Firestore yet. Run at least one session.")
    else:
        # ── Top KPIs ──────────────────────────────────
        total_sessions = df_all["session_id"].nunique()
        total_records  = len(df_all)
        overall_present= (df_all["status"] == "Present").sum()
        overall_late   = (df_all["status"] == "Late").sum()
        overall_absent = (df_all["status"] == "Absent").sum()
        attendance_rate = round(
            (overall_present + overall_late) / total_records * 100, 1
        ) if total_records else 0

        st.markdown(
            """<div style="font-family:'Orbitron',monospace;font-size:0.7rem;
                color:#00e5ff;letter-spacing:0.22em;text-transform:uppercase;
                margin-bottom:12px;">Global Metrics</div>""",
            unsafe_allow_html=True,
        )
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Sessions",       total_sessions)
        k2.metric("Records",        total_records)
        k3.metric("Present",        overall_present)
        k4.metric("Late",           overall_late)
        k5.metric("Absent",         overall_absent)
        k6.metric("Attend. Rate",   f"{attendance_rate}%")

        st.markdown("<hr style='border-color:rgba(0,229,255,0.1);margin:20px 0;'>", unsafe_allow_html=True)

        # ── Row 1: Donut + Bar ────────────────────────
        r1c1, r1c2 = st.columns([1, 2])

        with r1c1:
            st.markdown(
                """<div style="font-family:'Orbitron',monospace;font-size:0.65rem;
                    color:#00e5ff;letter-spacing:0.18em;margin-bottom:8px;">
                    ◉ OVERALL STATUS DIST.</div>""",
                unsafe_allow_html=True,
            )
            fig_donut = go.Figure(
                go.Pie(
                    labels=["Present", "Late", "Absent"],
                    values=[overall_present, overall_late, overall_absent],
                    hole=0.62,
                    marker=dict(
                        colors=["#00ff9d", "#ffd600", "#ff3d5a"],
                        line=dict(color="#040b14", width=3),
                    ),
                    textfont=dict(family="Rajdhani", color="#000000", size=13),
                    pull=[0.03, 0, 0],
                )
            )
            fig_donut.update_layout(
                **PLOTLY_LAYOUT,
                height=280,
                showlegend=True,
                annotations=[
                    dict(text=f"{attendance_rate}%<br><span style='font-size:9px'>ATTEND.</span>",
                         x=0.5, y=0.5,
                         font=dict(family="Orbitron", size=15, color="#00e5ff"),
                         showarrow=False)
                ],
            )
            fig_donut.update_layout(legend=dict(font=dict(color="#00e5ff")))
            st.plotly_chart(fig_donut, use_container_width=True)

        with r1c2:
            st.markdown(
                """<div style="font-family:'Orbitron',monospace;font-size:0.65rem;
                    color:#00e5ff;letter-spacing:0.18em;margin-bottom:8px;">
                    ◉ ATTENDANCE BY SESSION</div>""",
                unsafe_allow_html=True,
            )
            sess_grp = (
                df_all.groupby(["session_id", "status"])
                .size()
                .reset_index(name="count")
            )
            fig_bar = go.Figure()
            color_map = {"Present": "#00ff9d", "Late": "#ffd600", "Absent": "#ff3d5a"}
            for status, color in color_map.items():
                sub = sess_grp[sess_grp["status"] == status]
                fig_bar.add_trace(
                    go.Bar(
                        name=status,
                        x=sub["session_id"],
                        y=sub["count"],
                        marker_color=color,
                        marker_line_color="#040b14",
                        marker_line_width=1,
                        opacity=0.88,
                    )
                )
            fig_bar.update_layout(**PLOTLY_LAYOUT)
            fig_bar.update_layout(
                barmode="stack",
                height=280,
                xaxis_tickangle=-30,
                xaxis=dict(tickfont=dict(family="Share Tech Mono", size=10),
                           gridcolor="rgba(0,229,255,0.05)"),
                yaxis=dict(gridcolor="rgba(0,229,255,0.05)"),
            )
            fig_bar.update_layout(legend=dict(font=dict(color="#00e5ff")))
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Row 2: Line trend + Heatmap ───────────────
        r2c1, r2c2 = st.columns(2)

        with r2c1:
            st.markdown(
                """<div style="font-family:'Orbitron',monospace;font-size:0.65rem;
                    color:#00e5ff;letter-spacing:0.18em;margin-bottom:8px;">
                    ◉ DAILY ATTENDANCE TREND</div>""",
                unsafe_allow_html=True,
            )
            df_ts = df_all.copy()
            df_ts["date"] = pd.to_datetime(df_ts["timestamp"]).dt.date
            daily = (
                df_ts.groupby(["date", "status"])
                .size()
                .reset_index(name="count")
            )
            fig_line = go.Figure()
            for status, color in color_map.items():
                sub = daily[daily["status"] == status].sort_values("date")
                fig_line.add_trace(
                    go.Scatter(
                        name=status,
                        x=sub["date"],
                        y=sub["count"],
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        marker=dict(size=6, color=color,
                                    line=dict(color="#040b14", width=1)),
                        fill="tozeroy",
                        fillcolor=color.replace("#", "rgba(") + ",0.06)".replace("rgba(#", "rgba(")
                        if False else "rgba(0,0,0,0)",
                    )
                )
            fig_line.update_layout(**PLOTLY_LAYOUT, height=280)
            fig_line.update_layout(legend=dict(font=dict(color="#00e5ff")))
            st.plotly_chart(fig_line, use_container_width=True)

        with r2c2:
            st.markdown(
                """<div style="font-family:'Orbitron',monospace;font-size:0.65rem;
                    color:#00e5ff;letter-spacing:0.18em;margin-bottom:8px;">
                    ◉ ATTENDANCE RATE BY COURSE</div>""",
                unsafe_allow_html=True,
            )
            course_grp = df_all.groupby("course_code").apply(
                lambda g: round((g["status"].isin(["Present", "Late"]).sum()) / len(g) * 100, 1)
            ).reset_index(name="rate")
            fig_hbar = go.Figure(
                go.Bar(
                    y=course_grp["course_code"],
                    x=course_grp["rate"],
                    orientation="h",
                    marker=dict(
                        color=course_grp["rate"],
                        colorscale=[[0, "#ff3d5a"], [0.5, "#ffd600"], [1, "#00ff9d"]],
                        showscale=False,
                        line=dict(color="#040b14", width=1),
                    ),
                    text=[f"{r}%" for r in course_grp["rate"]],
                    textposition="outside",
                    textfont=dict(family="Share Tech Mono", color="#00e5ff", size=11),
                )
            )
            fig_hbar.update_layout(**PLOTLY_LAYOUT)
            fig_hbar.update_layout(
                height=280,
                xaxis=dict(range=[0, 115], gridcolor="rgba(0,229,255,0.05)"),
                yaxis=dict(gridcolor="rgba(0,229,255,0.05)"),
            )
            st.plotly_chart(fig_hbar, use_container_width=True)