import os
import cv2
import numpy as np
import time
import sqlite3
from datetime import datetime
from insightface.app import FaceAnalysis

FACES_DIR = 'faces'
EMBEDDINGS_DIR = 'embeddings'
DB_PATH = 'logs.db'

# Load all embeddings and names/roles
known_embeddings = []
known_names = []
known_roles = []
for file in os.listdir(EMBEDDINGS_DIR):
    if file.endswith('.npy'):
        embedding = np.load(os.path.join(EMBEDDINGS_DIR, file))
        name_role = os.path.splitext(file)[0]
        if '_' in name_role:
            name, role = name_role.rsplit('_', 1)
            known_names.append(name.replace('_', ' ').title())
            known_roles.append(role)
            known_embeddings.append(embedding)

# Setup database
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    name TEXT,
    role TEXT,
    event TEXT
)''')
conn.commit()

# Initialize InsightFace
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# Open video stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('❌ Could not open webcam.')
    exit(1)

last_seen = {}
LOG_INTERVAL = 10  # seconds

print('🟢 Recognition started. Press q to quit.')

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    faces = app.get(frame)
    current_names = set()
    for face in faces:
        embedding = face.normed_embedding
        dists = np.linalg.norm(np.array(known_embeddings) - embedding, axis=1)
        min_idx = np.argmin(dists)
        if dists[min_idx] < 0.8:  # Threshold for ArcFace
            name = known_names[min_idx]
            role = known_roles[min_idx]
            current_names.add(name)
            now = time.time()
            if name not in last_seen or now - last_seen[name] > LOG_INTERVAL:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                c.execute('INSERT INTO logs (timestamp, name, role, event) VALUES (?, ?, ?, ?)',
                          (timestamp, name, role, 'present'))
                conn.commit()
                print(f'✅ {timestamp} - {name} ({role}) present')
                last_seen[name] = now
            # Draw box and label
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
            cv2.putText(frame, f'{name} ({role})', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow('Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close() 