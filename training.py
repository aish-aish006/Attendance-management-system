# training.py
import cv2
import os
import numpy as np

dataset_path = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faces = []
labels = []
label_map = {}

for label_id, person in enumerate(os.listdir(dataset_path)):
    person_path = os.path.join(dataset_path, person)
    label_map[label_id] = person

    for image_file in os.listdir(person_path):
        image_path = os.path.join(person_path, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        detected_faces = face_cascade.detectMultiScale(img)
        for (x, y, w, h) in detected_faces:
            faces.append(img[y:y+h, x:x+w])
            labels.append(label_id)

recognizer.train(faces, np.array(labels))
recognizer.save("face_recognition.yml")

# Save label map
import pickle
with open("labels.pickle", "wb") as f:
    pickle.dump(label_map, f)

print("[INFO] Training complete and model saved.")
