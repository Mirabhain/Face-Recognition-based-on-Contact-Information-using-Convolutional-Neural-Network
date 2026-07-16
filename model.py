import os
import numpy as np
import sqlite3
from deepface import DeepFace

DB_PATH = r'C:\Users\hp\Downloads\facerecognition\database\faces.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS contacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    phone TEXT,
                    email TEXT,
                    embedding TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()

def register_face(image_path, name, phone, email):
    try:
        embedding_obj = DeepFace.represent(
            img_path=image_path,
            model_name='Facenet',
            enforce_detection=False
        )
        embedding = embedding_obj[0]['embedding']
        embedding_str = ','.join(map(str, embedding))
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO contacts (name, phone, email, embedding) VALUES (?, ?, ?, ?)",
                  (name, phone, email, embedding_str))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def recognize_face(image_path):
    try:
        embedding_obj = DeepFace.represent(
            img_path=image_path,
            model_name='Facenet',
            enforce_detection=False
        )
        input_embedding = np.array(embedding_obj[0]['embedding'])

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name, phone, email, embedding FROM contacts")
        rows = c.fetchall()
        conn.close()

        if not rows:
            return None, 0

        best_match = None
        best_score = -1

        for name, phone, email, emb_str in rows:
            stored_emb = np.array(list(map(float, emb_str.split(','))))
            similarity = np.dot(input_embedding, stored_emb) / (
                np.linalg.norm(input_embedding) * np.linalg.norm(stored_emb))
            if similarity > best_score:
                best_score = similarity
                best_match = {'name': name, 'phone': phone, 'email': email}

        if best_score > 0.7:
            return best_match, round(best_score * 100, 2)
        else:
            return None, round(best_score * 100, 2)

    except Exception as e:
        print(f"Error: {e}")
        return None, 0