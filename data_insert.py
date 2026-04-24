from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    cred = credentials.Certificate("firebase/serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()
students = [
    ("222-16-657", "Mashrur Kabir"),
    ("222-16-662", "Sadman Chowdhury"),
    ("222-16-665", "Maisha Rahman"),
    ("222-16-675", "Nabila Sarkar"),
    ("222-16-677", "Saima Jahan"),
]

# 👇 Add multiple courses
courses = [
    "Software Development",
    "Web Development",
    "Machine Learning",
    "Database Systems",
    "Computer Networks"
]

data = [
    ["Present","Late","Present","Absent","Present"],
    ["Present","Present","Late","Present","Absent"],
    ["Late","Present","Present","Present","Present"],
    ["Absent","Present","Late","Present","Late"],
    ["Present","Present","Present","Late","Present"],
    ["Late","Absent","Present","Present","Present"],
    ["Present","Present","Absent","Late","Present"],
    ["Present","Late","Present","Present","Late"],
    ["Present","Present","Late","Absent","Present"],
    ["Present","Late","Present","Present","Late"],
]

base_date = datetime(2026, 4, 15, 10, 0)

for i, day in enumerate(data):
    session_date = base_date + timedelta(days=i)

    # 👇 Rotate course names
    course_name = courses[i % len(courses)]

    session_id = f"{course_name}_{session_date.strftime('%Y-%m-%d_%H-%M')}"

    db.collection("attendance").document(session_id).set({
        "course_code": course_name,   # ✅ now meaningful
        "session_id": session_id,
        "start_time": session_date,
        "late_deadline": session_date + timedelta(minutes=2),
        "created_at": firestore.SERVER_TIMESTAMP
    })

    for j, status in enumerate(day):
        sid, name = students[j]

        ts = session_date
        if status == "Late":
            ts += timedelta(minutes=5)

        db.collection("attendance").document(session_id)\
          .collection("students").document(sid).set({
              "name": name,
              "status": status,
              "timestamp": ts
          })

print("✅ Fake multi-course data inserted successfully!")