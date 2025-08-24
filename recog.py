import cv2
import os
import csv
from datetime import datetime

# Load trained recognizer and face cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognition.yml")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load names from dataset folder
def get_names():
    names = {}
    dataset_path = "dataset"
    for i, folder in enumerate(os.listdir(dataset_path)):
        names[i] = folder
    return names

names = get_names()

# Attendance file for today
today = datetime.now().strftime("%Y-%m-%d")
filename = f"attendance_{today}.csv"

# Create file with header if it doesn't exist
if not os.path.isfile(filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Name", "Time"])

# Mark attendance only once per person
def mark_attendance(id_, name):
    filename = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"

    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Name", "Time"])

    now = datetime.now().strftime("%H:%M:%S")
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([id_, name, now])
        print(f"[INFO] Attendance marked for {name} ({id_}) at {now}")
        return True



# Recognition threshold
recognition_threshold = 70  # Lower = more strict (good range: 50–70)

# Start camera
# Start camera
cap = cv2.VideoCapture(0)
print("[INFO] Starting face recognition...")

recognized_once = False  # Flag to track successful recognition

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera not detected.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        id_, confidence = recognizer.predict(roi_gray)
        print(f"[DEBUG] ID: {id_}, Confidence: {confidence:.2f}")

        if confidence < recognition_threshold:
            name = names.get(id_, "Unknown")
            color = (0, 255, 0)

            cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Ensure attendance is marked and THEN exit
            if not face_detected_and_marked:
                success = mark_attendance(id_, name)
                if success:
                    face_detected_and_marked = True

                else:
                    name = "Unknown"
                    color = (0, 0, 255)

        # Draw label and rectangle
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Face Recognition", frame)

    if recognized_once:
        print("[INFO] Recognition done. Exiting...")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quit key pressed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

