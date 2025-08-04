import os
import argparse
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np

FACES_DIR = 'faces'
EMBEDDINGS_DIR = 'embeddings'
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description='Register a new face')
parser.add_argument('--name', required=True, help='Person name')
parser.add_argument('--role', required=True, choices=['customer', 'employee', 'fraudster'], help='Role')
parser.add_argument('--image', required=True, help='Path to image file')
args = parser.parse_args()

# Load image
img = Image.open(args.image).convert('RGB')
img_np = np.array(img)

# Initialize InsightFace
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))
faces = app.get(img_np)

if not faces:
    print('❌ No face detected in the image.')
    exit(1)

face = faces[0]
embedding = face.normed_embedding

# Save image
safe_name = args.name.lower().replace(' ', '_')
img_save_path = os.path.join(FACES_DIR, f'{safe_name}_{args.role}.jpg')
img.save(img_save_path)

# Save embedding
embedding_save_path = os.path.join(EMBEDDINGS_DIR, f'{safe_name}_{args.role}.npy')
np.save(embedding_save_path, embedding)

print(f'✅ Registered {args.name} as {args.role}.') 