Step 1: Install dependencies
pip install -r requirements.txt

Step 2: Enroll employees (repeat for all)
python scripts/enroll_student.py

Step 3: Train model
python scripts/train_model.py

Step 4: Start a meeting session
python main_app.py

It will ask:
--> Meeting Title
--> Duration minutes
--> Selected employee IDs

A) Start Meeting Streamlit App

streamlit run dashboard.py
