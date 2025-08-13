import streamlit as st
import pandas as pd
import os
import sqlite3
from PIL import Image
import numpy as np
import cv2
import tempfile
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import io

# Streamlit Cloud optimizations
st.set_page_config(
    page_title='Face Recognition Dashboard', 
    layout='wide',
    initial_sidebar_state='expanded'
)

# Warning for Streamlit Cloud limitations
st.warning("⚠️ **Demo Version**: This is optimized for Streamlit Cloud's free tier. Some features may be limited due to resource constraints.")

DB_PATH = 'logs.db'
FACES_DIR = 'faces'
EMBEDDINGS_DIR = 'embeddings'
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

st.title('🧠 Face Recognition & Logging Dashboard (Demo)')

# Simplified model loading for Streamlit Cloud
@st.cache_resource
def load_face_analysis_model():
    """Load face analysis model - simplified for Streamlit Cloud"""
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis()
        app.prepare(ctx_id=-1, det_size=(160, 160))  # CPU only, smaller size
        return app
    except Exception as e:
        st.error(f"Face recognition model failed to load: {e}")
        return None

@st.cache_data(ttl=120)
def load_face_embeddings():
    """Load face embeddings with caching"""
    known_embeddings = []
    known_names = []
    known_roles = []
    
    if not os.path.exists(EMBEDDINGS_DIR):
        return known_embeddings, known_names, known_roles
        
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith('.npy'):
            try:
                embedding = np.load(os.path.join(EMBEDDINGS_DIR, file))
                name_role = os.path.splitext(file)[0]
                if '_' in name_role:
                    name, role = name_role.rsplit('_', 1)
                    known_names.append(name.replace('_', ' ').title())
                    known_roles.append(role)
                    known_embeddings.append(embedding)
            except Exception as e:
                st.error(f"Error loading {file}: {e}")
    
    return known_embeddings, known_names, known_roles

def get_db_logs():
    """Get logs from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query('SELECT * FROM logs ORDER BY timestamp DESC LIMIT 1000', conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

def get_registered_faces():
    """Get list of registered faces"""
    faces = []
    if not os.path.exists(FACES_DIR):
        return faces
        
    files = [f for f in os.listdir(FACES_DIR) if f.endswith('.jpg')]
    for file in files:
        name_role = os.path.splitext(file)[0]
        if '_' in name_role:
            name, role = name_role.rsplit('_', 1)
            faces.append({
                'name': name.replace('_', ' ').title(),
                'role': role,
                'img_path': os.path.join(FACES_DIR, file),
                'filename': file
            })
    return faces

def save_face_and_embedding(img, name, role):
    """Save face image and embedding - simplified for Streamlit Cloud"""
    app = load_face_analysis_model()
    if app is None:
        return False, 'Face recognition model not available'
    
    try:
        img_np = np.array(img.convert('RGB'))
        faces = app.get(img_np)
        if not faces:
            return False, 'No face detected.'
        
        face = faces[0]
        embedding = face.normed_embedding
        safe_name = name.lower().replace(' ', '_')
        
        # Save image
        img_save_path = os.path.join(FACES_DIR, f'{safe_name}_{role}.jpg')
        img.save(img_save_path)
        
        # Save embedding
        embedding_save_path = os.path.join(EMBEDDINGS_DIR, f'{safe_name}_{role}.npy')
        np.save(embedding_save_path, embedding)
        
        return True, f'Registered {name} as {role}.'
    except Exception as e:
        return False, f'Error: {str(e)}'

# Initialize database
def init_database():
    """Initialize SQLite database"""
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
    conn.close()

init_database()

# Sidebar Navigation
page = st.sidebar.radio('Navigate', [
    'Register Face', 
    'View Faces & Logs', 
    'Simple Recognition',
    'Analytics'
])

# Register Face Page
if page == 'Register Face':
    st.header('📸 Register a New Face')
    
    with st.form('register_form', clear_on_submit=True):
        name = st.text_input('Name')
        role = st.selectbox('Role', ['customer', 'employee', 'fraudster'])
        img_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
        
        submitted = st.form_submit_button('Register')
        
        if submitted:
            if not name or not img_file:
                st.error('Please provide both name and image.')
            else:
                img = Image.open(img_file)
                success, msg = save_face_and_embedding(img, name, role)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
    
    # Show registered faces
    st.subheader('Registered Faces')
    registered_faces = get_registered_faces()
    
    if registered_faces:
        cols = st.columns(3)
        for i, face in enumerate(registered_faces):
            with cols[i % 3]:
                st.image(face['img_path'], width=150, caption=f"{face['name']} ({face['role']})")
    else:
        st.info('No faces registered yet.')

# View Faces & Logs Page
elif page == 'View Faces & Logs':
    st.header('👥 Registered Faces')
    
    registered_faces = get_registered_faces()
    if registered_faces:
        for face in registered_faces:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(face['img_path'], width=120)
            with col2:
                st.write(f"**Name:** {face['name']}")
                st.write(f"**Role:** {face['role']}")
    else:
        st.info('No faces registered.')
    
    st.header('📊 Recent Logs')
    df = get_db_logs()
    
    if not df.empty:
        st.dataframe(df.head(50), use_container_width=True)
        
        # Simple stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Total Logs', len(df))
        with col2:
            st.metric('Unique People', df['name'].nunique() if 'name' in df.columns else 0)
        with col3:
            if 'role' in df.columns:
                fraudster_count = len(df[df['role'] == 'fraudster'])
                st.metric('Fraudster Detections', fraudster_count)
    else:
        st.info('No logs found.')

# Simple Recognition Page
elif page == 'Simple Recognition':
    st.header('🎥 Simple Face Recognition')
    st.info('📝 **Note**: Real-time recognition is limited on Streamlit Cloud. Upload a single image for recognition instead.')
    
    uploaded_img = st.file_uploader('Upload image for recognition', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_img:
        img = Image.open(uploaded_img)
        st.image(img, width=300, caption='Uploaded Image')
        
        if st.button('Recognize Faces'):
            app = load_face_analysis_model()
            if app is None:
                st.error('Face recognition model not available')
            else:
                known_embeddings, known_names, known_roles = load_face_embeddings()
                
                if not known_embeddings:
                    st.warning('No registered faces found. Please register faces first.')
                else:
                    img_np = np.array(img.convert('RGB'))
                    faces = app.get(img_np)
                    
                    if faces:
                        for i, face in enumerate(faces):
                            embedding = face.normed_embedding
                            dists = np.linalg.norm(np.array(known_embeddings) - embedding, axis=1)
                            min_idx = np.argmin(dists)
                            
                            if dists[min_idx] < 0.8:
                                name = known_names[min_idx]
                                role = known_roles[min_idx]
                                confidence = 1 - dists[min_idx]
                                
                                st.success(f'✅ **Recognized**: {name} ({role}) - Confidence: {confidence:.2f}')
                                
                                # Log the recognition
                                conn = sqlite3.connect(DB_PATH)
                                c = conn.cursor()
                                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                c.execute('INSERT INTO logs (timestamp, name, role, event) VALUES (?, ?, ?, ?)',
                                         (timestamp, name, role, 'image_recognition'))
                                conn.commit()
                                conn.close()
                            else:
                                st.warning(f'❓ Face {i+1}: Unknown person (Distance: {dists[min_idx]:.3f})')
                    else:
                        st.warning('No faces detected in the image.')

# Analytics Page
elif page == 'Analytics':
    st.header('📊 Analytics')
    
    df = get_db_logs()
    
    if not df.empty and len(df) > 0:
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Simple charts
        if 'role' in df.columns:
            role_counts = df['role'].value_counts()
            fig_pie = px.pie(values=role_counts.values, names=role_counts.index, title='Detections by Role')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        if 'name' in df.columns:
            daily_counts = df.groupby('date').size().reset_index(name='detections')
            fig_line = px.line(daily_counts, x='date', y='detections', title='Daily Detection Trend')
            st.plotly_chart(fig_line, use_container_width=True)
        
        # Export data
        st.subheader('📤 Export Data')
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f'face_recognition_logs_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )
    else:
        st.info('No data available for analytics.')

# Footer
st.markdown('---')
st.markdown('**Note**: This is a demo version optimized for Streamlit Community Cloud free tier. For full features, consider upgrading to a more powerful hosting solution.')