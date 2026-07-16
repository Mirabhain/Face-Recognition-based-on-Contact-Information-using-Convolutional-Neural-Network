from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import sqlite3
import base64
import os
from deepface import DeepFace

app = Flask(__name__)

DB_PATH = r'C:\Users\hp\Downloads\facerecognition\database\faces.db'
TMP_FACE = r'C:\Users\hp\Downloads\facerecognition\tmp_face.jpg'
TMP_REG = r'C:\Users\hp\Downloads\facerecognition\tmp_register.jpg'

camera = None
current_match = {'name': '', 'phone': '', 'email': '', 'confidence': 0, 'detected': False}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Pre-load model on startup
print("Loading FaceNet model... please wait...")
DeepFace.build_model("Facenet")
print("Model loaded!")

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

def get_all_embeddings():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, phone, email, embedding FROM contacts")
    rows = c.fetchall()
    conn.close()
    return rows

def get_embedding(img_path):
    result = DeepFace.represent(
        img_path=img_path,
        model_name='Facenet',
        enforce_detection=False
    )
    return np.array(result[0]['embedding'])

def cosine_match(input_emb):
    rows = get_all_embeddings()
    if not rows:
        return None, 0
    best_match = None
    best_score = -1
    for name, phone, email, emb_str in rows:
        stored = np.array(list(map(float, emb_str.split(','))))
        score = np.dot(input_emb, stored) / (np.linalg.norm(input_emb) * np.linalg.norm(stored))
        if score > best_score:
            best_score = score
            best_match = {'name': name, 'phone': phone, 'email': email}
    if best_score > 0.65:
        return best_match, round(best_score * 100, 2)
    return None, round(best_score * 100, 2)

def generate_frames():
    global camera, current_match
    camera = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

        for (x, y, w, h) in faces:
            if frame_count % 20 == 0:
                try:
                    face_crop = frame[y:y+h, x:x+w]
                    cv2.imwrite(TMP_FACE, face_crop)
                    input_emb = get_embedding(TMP_FACE)
                    match, confidence = cosine_match(input_emb)
                    if match:
                        current_match = {**match, 'confidence': confidence, 'detected': True}
                    else:
                        current_match = {'name': 'Unknown', 'phone': '-', 'email': '-',
                                        'confidence': confidence, 'detected': False}
                except Exception as e:
                    print(f"Recognition error: {e}")

            color = (0, 255, 0) if current_match.get('detected') else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = current_match.get('name', 'Detecting...')
            cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
            cv2.putText(frame, label, (x+5, y-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frame_count += 1
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_match')
def get_match():
    return jsonify(current_match)

@app.route('/get_contacts')
def get_contacts():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT DISTINCT name, phone, email FROM contacts")
    rows = c.fetchall()
    conn.close()
    return jsonify([{'name': r[0], 'phone': r[1], 'email': r[2]} for r in rows])

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({'success': True})

@app.route('/capture_frame')
def capture_frame():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    for _ in range(5):
        camera.read()
    ret, frame = camera.read()
    if ret:
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'success': True, 'image': img_base64})
    return jsonify({'success': False})

@app.route('/register_person', methods=['POST'])
def register_person():
    init_db()
    name = request.form.get('name', '').strip()
    phone = request.form.get('phone', '').strip()
    email = request.form.get('email', '').strip()
    file = request.files.get('photo')

    if not name or not file:
        return jsonify({'success': False, 'message': 'Missing name or photo'})

    file.save(TMP_REG)
    try:
        emb = get_embedding(TMP_REG)
        emb_str = ','.join(map(str, emb))
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO contacts (name, phone, email, embedding) VALUES (?, ?, ?, ?)",
                    (name, phone, email, emb_str))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': f'{name} registered!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/register_batch', methods=['POST'])
def register_batch():
    init_db()
    name = request.form.get('name', '').strip()
    phone = request.form.get('phone', '').strip()
    email = request.form.get('email', '').strip()
    files = request.files.getlist('photos')

    if not name or not files:
        return jsonify({'success': False, 'message': 'Missing name or photos'})

    success_count = 0
    for file in files:
        tmp = r'C:\Users\hp\Downloads\facerecognition\tmp_register.jpg'
        file.save(tmp)
        try:
            emb = get_embedding(tmp)
            emb_str = ','.join(map(str, emb))
            conn = sqlite3.connect(DB_PATH)
            conn.execute("INSERT INTO contacts (name, phone, email, embedding) VALUES (?, ?, ?, ?)",
                        (name, phone, email, emb_str))
            conn.commit()
            conn.close()
            success_count += 1
        except Exception as e:
            print(f"Error on image: {e}")

    return jsonify({
        'success': True,
        'message': f'{success_count}/{len(files)} images registered for {name}!'
    })

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    file = request.files.get('photo')
    if not file:
        return jsonify({'success': False, 'message': 'No photo'})
    tmp = r'C:\Users\hp\Downloads\facerecognition\tmp_upload.jpg'
    file.save(tmp)
    try:
        emb = get_embedding(tmp)
        match, confidence = cosine_match(emb)
        if match:
            return jsonify({'success': True, 'detected': True, **match, 'confidence': confidence})
        return jsonify({'success': True, 'detected': False, 'name': 'Unknown', 'confidence': confidence})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    init_db()
    app.run(debug=False, port=5500)