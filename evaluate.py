import os
import numpy as np
import sqlite3
from deepface import DeepFace
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

DB_PATH = r'C:\Users\hp\Downloads\facerecognition\database\faces.db'
DATASET_PATH = r'C:\Users\hp\Downloads\facerecognition\dataset'
# Mapping: dataset folder name -> exact name registered in database
NAME_MAPPING = {
    "Darshni": "UthayaDarshni",
    "Raden_Salma": "Raden Salma",
    "Siti_Amirah": "Siti Amirah",
}

def get_all_embeddings():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM contacts")
    rows = c.fetchall()
    conn.close()
    return rows

def recognize(img_path):
    try:
        result = DeepFace.represent(
            img_path=img_path,
            model_name='Facenet',
            enforce_detection=False
        )
        input_emb = np.array(result[0]['embedding'])
        rows = get_all_embeddings()
        if not rows:
            return 'Unknown'
        best_name = 'Unknown'
        best_score = -1
        for name, emb_str in rows:
            stored = np.array(list(map(float, emb_str.split(','))))
            score = np.dot(input_emb, stored) / (
                np.linalg.norm(input_emb) * np.linalg.norm(stored))
            if score > best_score:
                best_score = score
                best_name = name
        if best_score > 0.65:
            return best_name
        return 'Unknown'
    except:
        return 'Unknown'

print("Loading FaceNet model...")
DeepFace.build_model("Facenet")
print("Model loaded! Starting evaluation...\n")

y_true = []
y_pred = []

for person in os.listdir(DATASET_PATH):
    person_folder = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_folder):
        continue

    images = [f for f in os.listdir(person_folder)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Testing {person} ({len(images)} images)...")

    for img_file in images:
        img_path = os.path.join(person_folder, img_file)
        predicted = recognize(img_path)

        # Match person folder name to registered name
        actual = NAME_MAPPING.get(person, person.replace('_', ' '))

        y_true.append(actual)
        y_pred.append(predicted)

        status = "✅" if predicted == actual else "❌"
        print(f"  {status} {img_file} → Predicted: {predicted}")

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Overall Accuracy: {accuracy*100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

print("\n🔢 Confusion Matrix:")
labels = sorted(set(y_true))
cm = confusion_matrix(y_true, y_pred, labels=labels)
print("Labels:", labels)
print(cm)

print("\n" + "="*60)
print("Done!")