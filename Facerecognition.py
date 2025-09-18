import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle
from pymongo import MongoClient

# Initialize detector and embedder
detector = MTCNN()
embedder = FaceNet()

# Load encodings from pickle file
encodings_file = r"C:\Users\Rithwika reddy\Desktop\Sem2_1files\FaceRecognition\project\server\multiface1.pkl"
with open(encodings_file, "rb") as f:
    data = pickle.load(f)
    known_encodings = np.array(data["encodings"])
    known_names = data["names"]

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["faces"]                # Your DB
collection = db["imagedocuments"]   # Your collection

def preprocess_face(face):
    """Resize and normalize face image for embedding."""
    face = cv2.resize(face, (160, 160))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def retrieve_person_data(rollno):
    """Retrieve the data associated with the recognized person from MongoDB."""
    person_data = collection.find_one({"qrData.rollNo": rollno})
    if person_data:
        qr_data = person_data.get("qrData", {})
        return {
            "rollNo": qr_data.get("rollNo", "N/A"),
            "name": qr_data.get("name", "N/A"),
            "fatherName": qr_data.get("fatherName", "N/A"),
            "department": qr_data.get("department", "N/A"),
            "contact": qr_data.get("contact", "N/A"),
            "images_count": len(person_data.get("images", [])),
        }
    else:
        return {"rollNo": rollno, "name": "Unknown"}

def recognize_faces(image_path, threshold=0.7):
    """Recognize faces from input image path."""
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb_image)
    recognized_details = []

    for face in faces:
        x, y, width, height = face["box"]
        x, y = max(0, x), max(0, y)

        face_roi = rgb_image[y:y+height, x:x+width]
        preprocessed_face = preprocess_face(face_roi)
        embedding = embedder.model.predict(preprocessed_face)[0]

        distances = np.linalg.norm(known_encodings - embedding, axis=1)
        min_distance = np.min(distances)

        if min_distance < threshold:
            rollno = known_names[np.argmin(distances)]
            person_data = retrieve_person_data(rollno)
            recognized_details.append(person_data)
        else:
            recognized_details.append({"rollNo": "Unknown", "name": "Unknown"})

    return recognized_details

# ------------ Run Example ------------
if __name__ == "__main__":
    image_path = r"C:\Users\Rithwika reddy\Pictures\Saved Pictures\1000078757.jpg"
    results = recognize_faces(image_path)

    print("Recognition Results:")
    for person in results:
        print(person)
