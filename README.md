# 🎓 A real-time employee count in a meeting

A **real-time employee count in a meeting** that uses **face recognition** to automatically detect and record employee during a meeting or seesion.

The system integrates **OpenCV, Deep Learning-based face recognition, Firebase Firestore, and Streamlit** to provide a **live dashboard with camera preview, real-time employee tracking, and automated CSV report generation**.

## ![Smart Meeting Room Banner](https://cdn.prod.website-files.com/66fe188853419b48e8fe3154/6705f904b2deb8870cea03e5_multi-camera-solutions-logitech-1.webp)

## 🚀 Features

### 🎥 Real-Time Meeting Employee Count with Face Recognition

- Detects and recognizes employee faces from a **live camera feed**
- Uses **face embeddings with deep learning**
- Displays recognized names in the camera preview

### 📊 Smart Meeting Record Dashboard

- Built with **Streamlit**
- Real-time employee counting updates
- No page refresh required

### ⏱ Meeting Control

- Create meeting sessions with:
  - Meeting code
  - Meeting duration
  - Late entry allowance
  - Selected employee

### 🟢 Automatic Status Detection

Employees are marked automatically as:

| Status  | Condition                     |
| ------- | ----------------------------- |
| Present | Detected before late deadline |
| Late    | Detected after late deadline  |
| Absent  | Not detected during meeting   |

### ☁️ Firebase Cloud Storage

Meeting Record is stored in **Google Firebase Firestore**.

### 📁 CSV Report Export

After the meeting ends, the system automatically generates:
`meeting_YYYY-MM-DD_MEETINGCODE.csv`

### 🛑 Meeting Control

- Start meeting
- Stop meeting manually
- Auto save Meeting data

### 📈 Live Meeting Record Analytics

Dashboard shows:

- Total Employees
- Present count
- Late count
- Live table updates

---

# 🛠 Technologies Used

| Technology         | Purpose                         |
| ------------------ | ------------------------------- |
| Python             | Core programming language       |
| OpenCV             | Camera capture & face detection |
| face_recognition   | Deep learning face recognition  |
| Streamlit          | Interactive dashboard UI        |
| Firebase Firestore | Cloud database                  |
| Pandas             | Data processing                 |
| Pickle             | Store face encodings            |

---

# 🤖 System Workflow ⚙️🦾

### This is the System Workflow of this Project

```
Employee Registration
        │
        ▼
Dataset Collection (OpenCV)
        │
        ▼
Face Embedding Training (Deep learning face embeddings)
        │
        ▼
encodings.pickle
        │
        ▼
Real-time Recognition
(OpenCV + Face Recognition)
        │
        ▼
Meeting Employee Count Logic
        │
        ▼
Firebase Firestore
        │
        ▼
    CSV Report
```

---

---

# 📂 Project Structure

```
Smart_Meeting_Room_Deep_learning/
│
├── .streamlit/
├── data/
│   ├── dataset/
│   │   ├── 222-1-662_Sadman_Chowdhury/
│   │   ├── 222-16-657_Mashrur_Kabir/
│   │   ├── 222-16-658_Ishita_Islam/
│   │   ├── 222-16-665_Maisha_Rahman/
│   │   ├── 222-16-675_Nabila_Sarkar/
│   │   ├── 222-16-677_Saima_jahan/
│   │   ├── 222-16-681_Saidur_Rahman_Polash/
│   │   ├── 222-16-683_Mahmudul_Hasan_Anabil/
│   │   └── 222-16-685_Kamrul_Hasan/
│   │
│   └── encodings.pickle
│
├── firebase/
│   └── serviceAccountKey.json
│
├── scripts/
│   ├── attendance_logic.py
│   ├── enroll_student.py
│   └── train_model.py
│
├── .gitignore
├── dashboard.py
├── main_app.py
├── README.md
└── requirements.txt
```

---

# 📦 Installation Guide

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/sami5671/Real_Time_Meeting_Employee_Count.git
cd Real_Time_Meeting_Employee_Count
```

# Install Dependencies

Create requirements.txt:

```bash
streamlit
opencv-python
face-recognition
firebase-admin
pandas
numpy
```

# Install requirements.txt

```bash
pip install -r requirements.txt
```

```bash
python scripts/enroll_student.py
```

```bash
python scripts/train_model.py
```

```bash
python main_app.py
```

```bash
streamlit run dashboard.py
```

```bash
cloudflared tunnel --url http://localhost:8501
```

# 🔥 Firebase Setup

Go to:
https://console.firebase.google.com

### Create a New Project

Enable:

```
Firestore Database
```

### Create Service Account Key

Download:

```
serviceAccountKey.json
```

Place it inside:

```
firebase/serviceAccountKey.json
```

# 📊 Firestore Database Structure

```
attendance
   │
   └── 2026-03-10_Meet_401
           │
           └── Employees
                 │
                 ├── 22101
                 │      name: John
                 │      status: Present
                 │      timestamp: ...
                 │
                 └── 22102
                        name: Rahim
                        status: Late
```

# 👨‍💻 Author

### Md. Sami Alam

Computer Science & Engineering
AI & Computer Vision Enthusiast
