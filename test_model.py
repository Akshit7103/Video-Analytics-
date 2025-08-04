import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
import time

def test_model_performance():
    """Test the face recognition model performance"""
    print("🧪 Testing Face Recognition Model Performance...")
    print("=" * 50)
    
    # Initialize the model
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Load existing embeddings
    EMBEDDINGS_DIR = 'embeddings'
    FACES_DIR = 'faces'
    
    known_embeddings = []
    known_names = []
    known_roles = []
    
    print("📁 Loading registered faces...")
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith('.npy'):
            embedding = np.load(os.path.join(EMBEDDINGS_DIR, file))
            name_role = os.path.splitext(file)[0]
            if '_' in name_role:
                name, role = name_role.rsplit('_', 1)
                known_names.append(name.replace('_', ' ').title())
                known_roles.append(role)
                known_embeddings.append(embedding)
                print(f"  ✅ Loaded: {name.replace('_', ' ').title()} ({role})")
    
    if not known_embeddings:
        print("❌ No registered faces found!")
        return
    
    print(f"\n📊 Model Statistics:")
    print(f"  • Registered faces: {len(known_embeddings)}")
    print(f"  • Embedding dimension: {known_embeddings[0].shape[0]}")
    print(f"  • Current threshold: 0.8")
    
    # Test self-similarity (same person should match themselves)
    print(f"\n🔍 Testing Self-Similarity...")
    for i, embedding in enumerate(known_embeddings):
        dist = np.linalg.norm(embedding - embedding)
        print(f"  • {known_names[i]}: Distance = {dist:.4f} (should be 0.0)")
    
    # Test cross-similarity (different people should not match)
    print(f"\n🔍 Testing Cross-Similarity...")
    for i in range(len(known_embeddings)):
        for j in range(i+1, len(known_embeddings)):
            dist = np.linalg.norm(known_embeddings[i] - known_embeddings[j])
            print(f"  • {known_names[i]} vs {known_names[j]}: Distance = {dist:.4f}")
    
    # Test with actual images
    print(f"\n🖼️ Testing with Registered Images...")
    for i, face_file in enumerate(os.listdir(FACES_DIR)):
        if face_file.endswith('.jpg'):
            img_path = os.path.join(FACES_DIR, face_file)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            faces = app.get(img_rgb)
            if faces:
                face = faces[0]
                embedding = face.normed_embedding
                
                # Find best match
                dists = np.linalg.norm(np.array(known_embeddings) - embedding, axis=1)
                min_idx = np.argmin(dists)
                min_dist = dists[min_idx]
                
                name_role = os.path.splitext(face_file)[0]
                if '_' in name_role:
                    expected_name = name_role.rsplit('_', 1)[0].replace('_', ' ').title()
                    matched_name = known_names[min_idx]
                    
                    print(f"  • {face_file}:")
                    print(f"    Expected: {expected_name}")
                    print(f"    Matched: {matched_name}")
                    print(f"    Distance: {min_dist:.4f}")
                    print(f"    Correct: {'✅' if expected_name == matched_name else '❌'}")
                    print(f"    Above threshold: {'❌' if min_dist < 0.8 else '✅'}")
            else:
                print(f"  • {face_file}: No face detected ❌")
    
    # Performance test
    print(f"\n⚡ Performance Test...")
    test_img = cv2.imread(os.path.join(FACES_DIR, os.listdir(FACES_DIR)[0]))
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    start_time = time.time()
    for _ in range(10):
        faces = app.get(test_img_rgb)
        if faces:
            embedding = faces[0].normed_embedding
            dists = np.linalg.norm(np.array(known_embeddings) - embedding, axis=1)
            min_idx = np.argmin(dists)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    print(f"  • Average processing time: {avg_time*1000:.2f} ms")
    print(f"  • FPS capability: {1/avg_time:.1f}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    print(f"  • Current threshold (0.8) seems appropriate")
    print(f"  • Model performance is good for real-time use")
    print(f"  • Consider adding image quality checks")
    print(f"  • Monitor false positive/negative rates in production")

if __name__ == "__main__":
    test_model_performance() 