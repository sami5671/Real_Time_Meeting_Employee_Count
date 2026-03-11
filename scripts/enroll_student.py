import os
import time
import cv2


def collect_data(student_id, student_name):

    path = f"data/dataset/{student_id}_{student_name}"

    if not os.path.exists(path):
        os.makedirs(path)

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Camera not accessible")
        return

    count = 0
    print(f"Capturing photos for {student_name}. Look at the camera...")

    while True:

        ret, frame = cam.read()

        if not ret:
            print("Failed to grab frame")
            break

        cv2.putText(
            frame,
            f"Images Captured: {count}/50",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Registering Student", frame)

        key = cv2.waitKey(1)

        if key % 256 == 32:  # SPACE key
            img_name = f"{path}/img_{count}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"Saved {img_name}")
            count += 1
            time.sleep(0.3)

        if count >= 50:
            break

        if key & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    print(f"Dataset created for {student_name}")


if __name__ == "__main__":

    sid = input("Enter Student ID: ")
    sname = input("Enter Student Name: ")

    collect_data(sid, sname)