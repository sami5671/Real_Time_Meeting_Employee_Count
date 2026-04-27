import csv
import json
import pickle
from datetime import datetime, timedelta

import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("firebase/serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

data = pickle.loads(open("data/encodings.pickle", "rb").read())

with open("meeting_config.json") as f:
    config = json.load(f)

course = config["course"]
students = config["students"]

start_time = datetime.strptime(config["start_time"], "%H:%M").replace(
    year=datetime.now().year,
    month=datetime.now().month,
    day=datetime.now().day,
)

duration = config["duration"]
late_minutes = config["late_minutes"]

end_time = start_time + timedelta(minutes=duration)
late_deadline = start_time + timedelta(minutes=late_minutes)

session_id = datetime.now().strftime("%Y-%m-%d") + "_" + course

attendance = {}

for s in students:
    sid, name = s.split("_", 1)

    attendance[sid] = {
        "name": name,
        "status": "Absent",
        "time": "",
    }

# Write session metadata
db.collection("attendance").document(session_id).set({
    "session_id": session_id,
    "course": course,
    "start_time": start_time,
    "end_time": end_time,
    "created_at": firestore.SERVER_TIMESTAMP
})


def mark_attendance(student_info):

    sid, name = student_info.split("_", 1)

    if sid not in attendance:
        return

    if attendance[sid]["status"] != "Absent":
        return

    now = datetime.now()

    if now <= late_deadline:
        status = "Present"
    else:
        status = "Late"

    attendance[sid]["status"] = status
    attendance[sid]["time"] = now.strftime("%H:%M:%S")

    db.collection("attendance").document(session_id).collection("students").document(
        sid
    ).set(
        {
            "name": name,
            "student_id": sid,
            "status": status,
            "timestamp": now,
        }
    )

    print(name, "->", status)


video_capture = cv2.VideoCapture(0)

print("Starting Recognition...")

while True:

    if datetime.now() > end_time:
        print("Meeting Finished")
        break

    ret, frame = video_capture.read()

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

    cv2.imshow("Smart Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()

csv_file = f"attendance_{session_id}.csv"

with open(csv_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["Student ID", "Name", "Status", "Time"])

    for sid, info in attendance.items():
        writer.writerow([sid, info["name"], info["status"], info["time"]])

print("CSV saved:", csv_file)