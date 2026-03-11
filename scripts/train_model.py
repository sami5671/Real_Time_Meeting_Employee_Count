import os
import pickle
import face_recognition


def train_embeddings():

    dataset_path = "data/dataset"

    known_encodings = []
    known_names = []

    for student_dir in os.listdir(dataset_path):

        student_path = os.path.join(dataset_path, student_dir)

        if not os.path.isdir(student_path):
            continue

        print(f"Processing: {student_dir}")

        for img_name in os.listdir(student_path):

            img_path = os.path.join(student_path, img_name)

            image = face_recognition.load_image_file(img_path)

            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(student_dir)

    data = {
        "encodings": known_encodings,
        "names": known_names,
    }

    with open("data/encodings.pickle", "wb") as f:
        f.write(pickle.dumps(data))

    print("Training complete. Encodings saved.")


if __name__ == "__main__":
    train_embeddings()