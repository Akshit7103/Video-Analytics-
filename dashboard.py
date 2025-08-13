import streamlit as st
import pandas as pd
import os
import sqlite3
from PIL import Image
import numpy as np
import cv2
from insightface.app import FaceAnalysis
import tempfile
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

DB_PATH = 'logs.db'
FACES_DIR = 'faces'
EMBEDDINGS_DIR = 'embeddings'
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

st.set_page_config(page_title='Face Recognition Dashboard', layout='wide')
st.title('🧠 Face Recognition & Logging Dashboard')

# --- Cached model initialization ---
@st.cache_resource
def load_face_analysis_model():
    """Load and cache the InsightFace model for reuse"""
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(224, 224))  # Further reduced for maximum speed
    return app

@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_face_embeddings():
    """Load and cache face embeddings for faster lookup"""
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
    return known_embeddings, known_names, known_roles

# --- Helper functions ---
def get_db_logs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM logs ORDER BY timestamp DESC', conn)
    conn.close()
    return df

def get_registered_faces():
    files = [f for f in os.listdir(FACES_DIR) if f.endswith('.jpg')]
    faces = []
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
    from insightface.app import FaceAnalysis
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))
    img_np = np.array(img.convert('RGB'))
    faces = app.get(img_np)
    if not faces:
        return False, 'No face detected.'
    face = faces[0]
    embedding = face.normed_embedding
    safe_name = name.lower().replace(' ', '_')
    img_save_path = os.path.join(FACES_DIR, f'{safe_name}_{role}.jpg')
    img.save(img_save_path)
    embedding_save_path = os.path.join(EMBEDDINGS_DIR, f'{safe_name}_{role}.npy')
    np.save(embedding_save_path, embedding)
    return True, f'Registered {name} as {role}.'

def update_face_info(old_filename, new_name, new_role, new_img=None):
    """Update face information and optionally the image"""
    try:
        # Parse old filename to get old name and role
        old_name_role = os.path.splitext(old_filename)[0]
        if '_' in old_name_role:
            old_name, old_role = old_name_role.rsplit('_', 1)
        else:
            return False, 'Invalid filename format'
        
        # Create new filename
        safe_new_name = new_name.lower().replace(' ', '_')
        new_filename = f'{safe_new_name}_{new_role}.jpg'
        new_embedding_filename = f'{safe_new_name}_{new_role}.npy'
        
        old_img_path = os.path.join(FACES_DIR, old_filename)
        old_embedding_path = os.path.join(EMBEDDINGS_DIR, f'{old_name}_{old_role}.npy')
        new_img_path = os.path.join(FACES_DIR, new_filename)
        new_embedding_path = os.path.join(EMBEDDINGS_DIR, new_embedding_filename)
        
        # If new image is provided, process it
        if new_img:
            app = FaceAnalysis()
            app.prepare(ctx_id=0, det_size=(640, 640))
            img_np = np.array(new_img.convert('RGB'))
            faces = app.get(img_np)
            if not faces:
                return False, 'No face detected in new image.'
            face = faces[0]
            embedding = face.normed_embedding
            new_img.save(new_img_path)
            np.save(new_embedding_path, embedding)
        else:
            # Just rename existing files
            if os.path.exists(old_img_path):
                os.rename(old_img_path, new_img_path)
            if os.path.exists(old_embedding_path):
                os.rename(old_embedding_path, new_embedding_path)
        
        # Update database logs if name changed
        if old_name.replace('_', ' ').title() != new_name:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('UPDATE logs SET name = ? WHERE name = ?', 
                     (new_name, old_name.replace('_', ' ').title()))
            conn.commit()
            conn.close()
        
        return True, f'Updated {new_name} as {new_role}.'
    except Exception as e:
        return False, f'Error updating face: {str(e)}'

def delete_face(filename):
    """Delete a face and its associated files"""
    try:
        name_role = os.path.splitext(filename)[0]
        if '_' in name_role:
            name, role = name_role.rsplit('_', 1)
        else:
            return False, 'Invalid filename format'
        
        # Delete image file
        img_path = os.path.join(FACES_DIR, filename)
        if os.path.exists(img_path):
            os.remove(img_path)
        
        # Delete embedding file
        embedding_path = os.path.join(EMBEDDINGS_DIR, f'{name}_{role}.npy')
        if os.path.exists(embedding_path):
            os.remove(embedding_path)
        
        # Remove from database logs
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM logs WHERE name = ?', (name.replace('_', ' ').title(),))
        conn.commit()
        conn.close()
        
        return True, f'Deleted {name.replace("_", " ").title()}.'
    except Exception as e:
        return False, f'Error deleting face: {str(e)}'

def get_filtered_logs(start_date=None, end_date=None, person=None, role=None):
    """Get filtered logs based on criteria"""
    conn = sqlite3.connect(DB_PATH)
    
    query = 'SELECT * FROM logs WHERE 1=1'
    params = []
    
    if start_date:
        query += ' AND date(timestamp) >= ?'
        params.append(start_date.strftime('%Y-%m-%d'))
    
    if end_date:
        query += ' AND date(timestamp) <= ?'
        params.append(end_date.strftime('%Y-%m-%d'))
    
    if person:
        query += ' AND name LIKE ?'
        params.append(f'%{person}%')
    
    if role:
        query += ' AND role = ?'
        params.append(role)
    
    query += ' ORDER BY timestamp DESC'
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def create_detection_charts(df):
    """Create real-time detection charts"""
    if df.empty:
        return None, None, None
    
    try:
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
    except Exception as e:
        st.error(f"Error processing timestamps: {e}")
        return None, None, None
    
    # 1. Daily Detection Trend
    daily_counts = df.groupby('date').size().reset_index(name='detections')
    daily_fig = px.line(daily_counts, x='date', y='detections', 
                       title='Daily Detection Trend',
                       labels={'detections': 'Number of Detections', 'date': 'Date'})
    daily_fig.update_layout(height=300)
    
    # 2. Hourly Activity Heatmap
    hourly_counts = df.groupby(['date', 'hour']).size().reset_index(name='detections')
    hourly_fig = px.scatter(hourly_counts, x='hour', y='date', size='detections',
                           title='Hourly Activity Heatmap',
                           labels={'hour': 'Hour of Day', 'date': 'Date', 'detections': 'Detections'})
    hourly_fig.update_layout(height=300)
    
    # 3. Role-based Detection Pie Chart
    role_counts = df['role'].value_counts()
    role_fig = px.pie(values=role_counts.values, names=role_counts.index,
                     title='Detections by Role')
    role_fig.update_layout(height=300)
    
    return daily_fig, hourly_fig, role_fig

def create_person_activity_chart(df):
    """Create person-specific activity chart"""
    if df.empty:
        return None
    
    try:
        # Ensure timestamp is datetime
        if 'timestamp' not in df.columns or df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by person and date
        person_daily = df.groupby(['name', 'date']).size().reset_index(name='detections')
    except Exception as e:
        st.error(f"Error creating person activity chart: {e}")
        return None
    
    # Create line chart for each person
    fig = px.line(person_daily, x='date', y='detections', color='name',
                  title='Person Activity Over Time',
                  labels={'detections': 'Number of Detections', 'date': 'Date'})
    fig.update_layout(height=400)
    
    return fig

def _process_frame_batch(batch_frames, app, known_embeddings_np, known_names, known_roles, detections, person_detections, fps):
    """Process a batch of frames efficiently"""
    batch_detection_count = 0
    
    for frame, frame_count in batch_frames:
        # Resize frame for faster processing while maintaining face detection quality
        height, width = frame.shape[:2]
        if width > 800:  # Only resize if frame is large
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Detect faces in frame
        faces = app.get(frame)
        
        for face in faces:
            embedding = face.normed_embedding
            
            # Vectorized distance calculation for all known embeddings at once
            if len(known_embeddings_np) > 0:
                dists = np.linalg.norm(known_embeddings_np - embedding, axis=1)
                min_idx = np.argmin(dists)
                
                if dists[min_idx] < 0.8:
                    name = known_names[min_idx]
                    role = known_roles[min_idx]
                    batch_detection_count += 1
                    
                    # Track detections per person
                    if name not in person_detections:
                        person_detections[name] = {
                            'count': 0,
                            'role': role,
                            'first_seen': frame_count / fps,
                            'last_seen': frame_count / fps
                        }
                    
                    person_detections[name]['count'] += 1
                    person_detections[name]['last_seen'] = frame_count / fps
                    
                    # Add to detections list
                    detections.append({
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'name': name,
                        'role': role
                    })
    
    return batch_detection_count

# --- Sidebar Navigation ---
page = st.sidebar.radio('Go to', ['Register Face', 'Edit Faces', 'Real-Time Recognition', 'Video Upload & Analysis', 'Analytics & Reports', 'View Logs & Faces'])

# --- Register Face ---
if page == 'Register Face':
    st.header('📸 Register a New Face')
    if 'show_camera' not in st.session_state:
        st.session_state['show_camera'] = False
    if 'last_registered' not in st.session_state:
        st.session_state['last_registered'] = 0
    with st.form('register_form', clear_on_submit=True):
        name = st.text_input('Name')
        role = st.selectbox('Role', ['customer', 'employee', 'fraudster'])
        st.markdown('**Choose one:**')
        img_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
        capture_btn = st.form_submit_button('Capture from Webcam')
        cam_img = None
        if capture_btn:
            st.session_state['show_camera'] = True
        if st.session_state['show_camera']:
            cam_img = st.camera_input('Take Photo and Submit')
        submitted = st.form_submit_button('Register')
        if submitted:
            if not name or (not img_file and not cam_img):
                st.error('Please provide a name and either upload or capture an image.')
            else:
                if img_file:
                    img = Image.open(img_file)
                else:
                    img = Image.open(cam_img)
                success, msg = save_face_and_embedding(img, name, role)
                if success:
                    st.success(msg)
                    st.session_state['show_camera'] = False
                    st.session_state['last_registered'] += 1  # trigger refresh
                else:
                    st.error(msg)
    st.markdown('---')
    st.subheader('Registered Faces')
    # Use last_registered to force refresh
    _ = st.session_state['last_registered']
    for face in get_registered_faces():
        st.write(f"{face['name']} ({face['role']})")
        st.image(face['img_path'], width=200)

# --- Edit Faces ---
elif page == 'Edit Faces':
    st.header('✏️ Edit Registered Faces')
    
    if 'editing_face' not in st.session_state:
        st.session_state['editing_face'] = None
    if 'show_edit_camera' not in st.session_state:
        st.session_state['show_edit_camera'] = False
    if 'last_edited' not in st.session_state:
        st.session_state['last_edited'] = 0
    
    registered_faces = get_registered_faces()
    
    if not registered_faces:
        st.info('No registered faces found. Register some faces first!')
    else:
        # Display all faces with edit options
        for i, face in enumerate(registered_faces):
            col1, col2, col3 = st.columns([2, 3, 1])
            
            with col1:
                st.image(face['img_path'], width=150, caption=f"Current: {face['name']} ({face['role']})")
            
            with col2:
                if st.button(f"Edit {face['name']}", key=f"edit_{i}"):
                    st.session_state['editing_face'] = face
                    st.session_state['show_edit_camera'] = False
            
            with col3:
                if st.button(f"Delete {face['name']}", key=f"delete_{i}", type="secondary"):
                    st.session_state['delete_confirmation'] = face
                    st.rerun()
        
        # Delete confirmation dialog
        if 'delete_confirmation' in st.session_state and st.session_state['delete_confirmation']:
            face_to_delete = st.session_state['delete_confirmation']
            st.markdown('---')
            st.subheader('⚠️ Confirm Deletion')
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(face_to_delete['img_path'], width=150, caption=f"Delete: {face_to_delete['name']} ({face_to_delete['role']})")
            
            with col2:
                st.warning(f"Are you sure you want to delete **{face_to_delete['name']}** ({face_to_delete['role']})?")
                st.write("This action will:")
                st.write("• Remove the face image and embedding")
                st.write("• Delete all associated log entries")
                st.write("• **This action cannot be undone!**")
                
                col_confirm1, col_confirm2, col_confirm3 = st.columns(3)
                with col_confirm1:
                    if st.button("✅ Yes, Delete", type="primary", key="confirm_delete"):
                        success, msg = delete_face(face_to_delete['filename'])
                        if success:
                            st.success(msg)
                            st.session_state['delete_confirmation'] = None
                            st.session_state['last_edited'] += 1
                            st.rerun()
                        else:
                            st.error(msg)
                
                with col_confirm2:
                    if st.button("❌ Cancel", key="cancel_delete"):
                        st.session_state['delete_confirmation'] = None
                        st.rerun()
                
                with col_confirm3:
                    if st.button("🔙 Go Back", key="back_delete"):
                        st.session_state['delete_confirmation'] = None
                        st.rerun()

        # Edit form
        if st.session_state['editing_face']:
            st.markdown('---')
            st.subheader(f"Editing: {st.session_state['editing_face']['name']}")
            
            with st.form('edit_form', clear_on_submit=False):
                new_name = st.text_input('Name', value=st.session_state['editing_face']['name'])
                new_role = st.selectbox('Role', ['customer', 'employee', 'fraudster'], 
                                      index=['customer', 'employee', 'fraudster'].index(st.session_state['editing_face']['role']))
                
                st.markdown('**Update photo (optional):**')
                new_img_file = st.file_uploader('Upload new image', type=['jpg', 'jpeg', 'png'], key='edit_upload')
                capture_new_btn = st.form_submit_button('Capture new photo')
                new_cam_img = None
                
                if capture_new_btn:
                    st.session_state['show_edit_camera'] = True
                
                if st.session_state['show_edit_camera']:
                    new_cam_img = st.camera_input('Take new photo', key='edit_camera')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    update_btn = st.form_submit_button('Update Face')
                with col2:
                    cancel_btn = st.form_submit_button('Cancel')
                with col3:
                    if st.form_submit_button('Keep current photo'):
                        # Check if any changes were made
                        original_face = st.session_state['editing_face']
                        if (new_name != original_face['name'] or new_role != original_face['role']):
                            # Update without changing photo
                            success, msg = update_face_info(
                                st.session_state['editing_face']['filename'], 
                                new_name, 
                                new_role
                            )
                            if success:
                                st.success(msg)
                                st.session_state['editing_face'] = None
                                st.session_state['show_edit_camera'] = False
                                st.session_state['last_edited'] += 1
                                st.rerun()
                            else:
                                st.error(msg)
                        else:
                            st.info("No changes detected. Please modify name or role before updating.")
                
                if update_btn:
                    if not new_name:
                        st.error('Please provide a name.')
                    elif not new_img_file and not new_cam_img:
                        st.error('Please provide a new image or choose "Keep current photo".')
                    else:
                        # Show confirmation for photo update
                        original_face = st.session_state['editing_face']
                        st.warning(f"⚠️ **Update Confirmation**")
                        st.write(f"You are about to update **{original_face['name']}** ({original_face['role']}) to:")
                        st.write(f"• **Name:** {new_name}")
                        st.write(f"• **Role:** {new_role}")
                        st.write(f"• **Photo:** {'New photo will be uploaded' if (new_img_file or new_cam_img) else 'Current photo will be kept'}")
                        st.write("**This will update the face recognition system and may affect existing logs.**")
                        
                        col_confirm1, col_confirm2 = st.columns(2)
                        with col_confirm1:
                            if st.button("✅ Confirm Update", type="primary", key="confirm_update"):
                                new_img = None
                                if new_img_file:
                                    new_img = Image.open(new_img_file)
                                elif new_cam_img:
                                    new_img = Image.open(new_cam_img)
                                
                                success, msg = update_face_info(
                                    st.session_state['editing_face']['filename'], 
                                    new_name, 
                                    new_role, 
                                    new_img
                                )
                                if success:
                                    st.success(msg)
                                    st.session_state['editing_face'] = None
                                    st.session_state['show_edit_camera'] = False
                                    st.session_state['last_edited'] += 1
                                    st.rerun()
                                else:
                                    st.error(msg)
                        
                        with col_confirm2:
                            if st.button("❌ Cancel Update", key="cancel_update"):
                                st.rerun()
                
                if cancel_btn:
                    st.session_state['editing_face'] = None
                    st.session_state['show_edit_camera'] = False
                    st.rerun()

# --- Real-Time Recognition ---
elif page == 'Real-Time Recognition':
    st.header('🎥 Real-Time Recognition')
    if 'recognizing' not in st.session_state:
        st.session_state['recognizing'] = False
    if 'stream_url' not in st.session_state:
        st.session_state['stream_url'] = "http://192.168.1.29:4747/video"

    if not st.session_state['recognizing']:
        st.subheader('📹 Video Source Configuration')
        
        # Video source selection
        video_source = st.radio(
            "Select video source:",
            ["Webcam (Default)", "IP Camera/Stream URL"],
            horizontal=True
        )
        
        if video_source == "Webcam (Default)":
            selected_source = 0
            st.info("Using default webcam (camera index 0)")
        else:
            # Stream URL input
            stream_url = st.text_input(
                "Enter stream URL:",
                value=st.session_state['stream_url'],
                placeholder="http://192.168.1.29:4747/video",
                help="Enter the full URL for your IP camera or stream (e.g., DroidCam, RTSP, HTTP stream)"
            )
            st.session_state['stream_url'] = stream_url
            selected_source = stream_url
            
            # Show some common examples
            with st.expander("📋 Common Stream URL Examples"):
                st.code("DroidCam: http://192.168.1.XX:4747/video")
                st.code("RTSP Camera: rtsp://username:password@192.168.1.XX:554/stream")
                st.code("HTTP Stream: http://192.168.1.XX:8080/stream")
                st.code("YouTube Live: https://www.youtube.com/watch?v=VIDEO_ID")
        
        if st.button('🚀 Start Recognition', type='primary'):
            if video_source == "IP Camera/Stream URL" and not stream_url.strip():
                st.error("Please enter a valid stream URL")
            else:
                st.session_state['recognizing'] = True
                st.session_state['selected_source'] = selected_source
                st.rerun()

    if st.session_state['recognizing']:
        stop = st.button('⏹️ Stop Recognition', key='unique_stop_btn', type='secondary')
        stframe = st.empty()
        
        # Display current video source
        source_info = st.empty()
        if st.session_state['selected_source'] == 0:
            source_info.info("📹 Using: Default Webcam")
        else:
            source_info.info(f"📹 Using: {st.session_state['selected_source']}")
        
        # Use cached model and embeddings
        app = load_face_analysis_model()
        known_embeddings, known_names, known_roles = load_face_embeddings()
        
        # Use the selected video source
        cap = cv2.VideoCapture(st.session_state['selected_source'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_count = 0
        last_seen = {}
        LOG_INTERVAL = 10
        SKIP_FRAMES = 8  # Further increased frame skipping for maximum performance
        
        # Batch database operations
        pending_logs = []
        last_batch_time = time.time()
        BATCH_SIZE = 5
        BATCH_INTERVAL = 3  # seconds
        
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
        # Performance monitoring
        processing_times = []
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while st.session_state['recognizing']:
                frame_start_time = time.time()
                ret, frame = cap.read()
                frame_count += 1
                
                # FPS calculation
                fps_counter += 1
                if fps_counter % 30 == 0:  # Update every 30 frames
                    fps = fps_counter / (time.time() - fps_start_time)
                    st.sidebar.metric("Live FPS", f"{fps:.1f}")
                    fps_counter = 0
                    fps_start_time = time.time()
                
                if frame_count % SKIP_FRAMES != 0:
                    if stop:
                        st.session_state['recognizing'] = False
                        break
                    continue  # Skip this frame for processing
                
                process_start = time.time()
                faces = app.get(frame)
                for face in faces:
                    embedding = face.normed_embedding
                    # Optimized distance calculation with early exit
                    min_dist = float('inf')
                    min_idx = -1
                    for i, known_emb in enumerate(known_embeddings):
                        dist = np.linalg.norm(known_emb - embedding)
                        if dist < min_dist:
                            min_dist = dist
                            min_idx = i
                            # Early exit if we find a very good match
                            if dist < 0.3:
                                break
                    
                    if min_idx != -1 and min_dist < 0.8:
                        name = known_names[min_idx]
                        role = known_roles[min_idx]
                        now = time.time()
                        if name not in last_seen or now - last_seen[name] > LOG_INTERVAL:
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            # Add to batch instead of immediate DB write
                            pending_logs.append((timestamp, name, role, 'present'))
                            last_seen[name] = now
                        bbox = face.bbox.astype(int)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
                        cv2.putText(frame, f'{name} ({role})', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                
                # Track processing time
                processing_time = time.time() - process_start
                processing_times.append(processing_time)
                if len(processing_times) > 10:  # Keep last 10 measurements
                    processing_times.pop(0)
                
                # Show processing stats in sidebar
                if len(processing_times) > 0:
                    avg_processing = sum(processing_times) / len(processing_times)
                    st.sidebar.metric("Avg Processing Time", f"{avg_processing*1000:.1f}ms")
                
                # Batch database operations
                current_time = time.time()
                if (len(pending_logs) >= BATCH_SIZE or 
                    current_time - last_batch_time > BATCH_INTERVAL) and pending_logs:
                    try:
                        c.executemany('INSERT INTO logs (timestamp, name, role, event) VALUES (?, ?, ?, ?)', 
                                    pending_logs)
                        conn.commit()
                        pending_logs.clear()
                        last_batch_time = current_time
                    except Exception as e:
                        st.error(f'Database error: {e}')
                
                stframe.image(frame, channels='BGR')
                if stop:
                    st.session_state['recognizing'] = False
                    break
        finally:
            # Flush any remaining logs
            if pending_logs:
                try:
                    c.executemany('INSERT INTO logs (timestamp, name, role, event) VALUES (?, ?, ?, ?)', 
                                pending_logs)
                    conn.commit()
                except Exception as e:
                    st.error(f'Final database flush error: {e}')
            cap.release()
            conn.close()
            cv2.destroyAllWindows()

# --- Video Upload & Analysis ---
elif page == 'Video Upload & Analysis':
    st.header('📹 Video Upload & Analysis')
    
    # File upload section
    st.subheader('📤 Upload Video File')
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: MP4, AVI, MOV, MKV (Max size: 200MB)"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
        st.info(f"📁 **File Info:** {uploaded_file.name} ({file_size:.2f} MB)")
        
        # Check file size
        if file_size > 200:
            st.error("❌ File size exceeds 200MB limit. Please upload a smaller file.")
        else:
            # Analysis options
            col1, col2 = st.columns(2)
            with col1:
                max_detections = st.number_input('Stop after X detections (0 = no limit)', min_value=0, value=0)
            with col2:
                max_duration = st.number_input('Max analysis time (minutes, 0 = no limit)', min_value=0, value=5)
            
            # Analysis button
            if st.button('🔍 Start Video Analysis', type='primary'):
                with st.spinner('🔄 Analyzing video... This may take a while depending on video length.'):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_video_path = tmp_file.name
                    
                    try:
                        # Load known faces
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
                        
                        # Initialize face analysis with optimized settings for video
                        app = FaceAnalysis()
                        app.prepare(ctx_id=0, det_size=(224, 224))  # Smaller detection size for speed
                        
                        # Open video file
                        cap = cv2.VideoCapture(temp_video_path)
                        if not cap.isOpened():
                            st.error("❌ Error: Could not open video file.")
                        else:
                            # Get video properties
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            duration = total_frames / fps if fps > 0 else 0
                            
                            # Analysis variables
                            frame_count = 0
                            detection_count = 0
                            detections = []
                            person_detections = {}
                            
                            # Optimized batch processing
                            batch_frames = []
                            batch_size = 32  # Process frames in batches
                            frame_skip = min(max(int(fps / 2), 10), 30)  # Dynamic frame sampling
                            
                            # Progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Convert known_embeddings to numpy array for vectorized operations
                            known_embeddings_np = np.array(known_embeddings) if known_embeddings else np.array([])
                            
                            # Early stopping variables
                            analysis_start_time = time.time()
                            max_duration_seconds = max_duration * 60 if max_duration > 0 else float('inf')
                            
                            # Process video frames
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    # Process remaining batch
                                    if batch_frames:
                                        detection_count += _process_frame_batch(batch_frames, app, known_embeddings_np, known_names, known_roles,
                                                                detections, person_detections, fps)
                                    break
                                
                                frame_count += 1
                                
                                # Early stopping checks
                                if max_detections > 0 and len(detections) >= max_detections:
                                    status_text.text(f"🛑 Stopped: Reached {max_detections} detections limit")
                                    break
                                
                                if time.time() - analysis_start_time > max_duration_seconds:
                                    status_text.text(f"🛑 Stopped: Reached {max_duration} minute time limit")
                                    break
                                
                                # Update progress every 50 frames for better responsiveness
                                if frame_count % 50 == 0 and total_frames > 0:
                                    progress = frame_count / total_frames
                                    progress_bar.progress(progress)
                                    elapsed_time = (time.time() - analysis_start_time) / 60
                                    status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%) - {len(detections)} detections - {elapsed_time:.1f}min elapsed")
                                
                                # Dynamic frame sampling
                                if frame_count % frame_skip != 0:
                                    continue
                                
                                # Add frame to batch
                                batch_frames.append((frame, frame_count))
                                
                                # Process batch when full
                                if len(batch_frames) >= batch_size:
                                    detection_count += _process_frame_batch(batch_frames, app, known_embeddings_np, 
                                                                               known_names, known_roles, detections, 
                                                                               person_detections, fps)
                                    batch_frames = []
                            
                            cap.release()
                        
                        # Clean up temporary file
                        os.unlink(temp_video_path)
                        
                        # Only show results if video was processed successfully
                        if 'detection_count' in locals():
                            # Display results
                            st.success(f"✅ Analysis complete! Found {detection_count} detections.")
                            
                            # Results section
                            st.subheader('📊 Analysis Results')
                            
                            # Summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric('Total Frames', total_frames)
                            
                            with col2:
                                st.metric('Video Duration', f"{duration:.1f}s")
                            
                            with col3:
                                st.metric('Total Detections', detection_count)
                            
                            with col4:
                                st.metric('Unique Persons', len(person_detections))
                            
                            # Person detection breakdown
                            if person_detections:
                                st.subheader('👥 Person Detection Breakdown')
                                
                                for name, data in person_detections.items():
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.write(f"**{name}**")
                                    
                                    with col2:
                                        st.write(f"Role: {data['role']}")
                                    
                                    with col3:
                                        st.write(f"Detections: {data['count']}")
                                    
                                    with col4:
                                        st.write(f"Duration: {data['last_seen'] - data['first_seen']:.1f}s")
                            
                            # Detailed detections table
                            if detections:
                                st.subheader('📋 Detailed Detections')
                                
                                # Create DataFrame for display
                                detections_df = pd.DataFrame(detections)
                                detections_df['timestamp_formatted'] = detections_df['timestamp'].apply(
                                    lambda x: f"{int(x//60):02d}:{int(x%60):02d}"
                                )
                                
                                # Display table
                                st.dataframe(
                                    detections_df[['frame', 'timestamp_formatted', 'name', 'role']].rename(
                                        columns={'timestamp_formatted': 'Time', 'name': 'Person', 'role': 'Role'}
                                    ),
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                # Export functionality
                                st.subheader('📤 Export Results')
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    csv_data = detections_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download CSV Report",
                                        data=csv_data,
                                        file_name=f'video_analysis_{uploaded_file.name.split(".")[0]}.csv',
                                        mime='text/csv'
                                    )
                                
                                with col2:
                                    # Create summary report
                                    summary_data = {
                                        'Metric': ['Total Frames', 'Video Duration (s)', 'Total Detections', 'Unique Persons'],
                                        'Value': [total_frames, f"{duration:.1f}", detection_count, len(person_detections)]
                                    }
                                    summary_df = pd.DataFrame(summary_data)
                                    summary_csv = summary_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download Summary",
                                        data=summary_csv,
                                        file_name=f'video_summary_{uploaded_file.name.split(".")[0]}.csv',
                                        mime='text/csv'
                                    )
                            
                            # Log detections to database
                            if detections:
                                conn = sqlite3.connect(DB_PATH)
                                c = conn.cursor()
                                
                                for detection in detections:
                                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    c.execute('INSERT INTO logs (timestamp, name, role, event) VALUES (?, ?, ?, ?)',
                                              (timestamp, detection['name'], detection['role'], 'video_analysis'))
                                
                                conn.commit()
                                conn.close()
                                
                                st.info("📝 Detections have been logged to the database.")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        # Clean up on error
                        if 'temp_video_path' in locals():
                            try:
                                os.unlink(temp_video_path)
                            except:
                                pass

# --- Analytics & Reports ---
elif page == 'Analytics & Reports':
    st.header('📊 Analytics & Reports')
    
    # Filter controls
    st.subheader('🔍 Filter Data')
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        start_date = st.date_input('Start Date', 
                                  value=(datetime.now() - timedelta(days=7)).date(),
                                  max_value=datetime.now().date())
    
    with col2:
        end_date = st.date_input('End Date', 
                                value=datetime.now().date(),
                                max_value=datetime.now().date())
    
    with col3:
        person_filter = st.text_input('Search Person', placeholder='Enter name...')
    
    with col4:
        role_filter = st.selectbox('Filter by Role', 
                                  ['All', 'customer', 'employee', 'fraudster'])
    
    # Apply filters
    role_filter = None if role_filter == 'All' else role_filter
    filtered_df = get_filtered_logs(start_date, end_date, person_filter, role_filter)
    
    # Summary statistics
    st.subheader('📈 Summary Statistics')
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_detections = len(filtered_df)
        st.metric('Total Detections', total_detections)
    
    with col2:
        unique_people = filtered_df['name'].nunique() if not filtered_df.empty else 0
        st.metric('Unique People', unique_people)
    
    with col3:
        fraudster_detections = len(filtered_df[filtered_df['role'] == 'fraudster']) if not filtered_df.empty else 0
        st.metric('Fraudster Detections', fraudster_detections, delta=fraudster_detections)
    
    with col4:
        if not filtered_df.empty:
            # Convert timestamp to datetime first
            filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
            avg_daily = filtered_df.groupby(filtered_df['timestamp'].dt.date).size().mean()
            st.metric('Avg Daily Detections', f'{avg_daily:.1f}')
        else:
            st.metric('Avg Daily Detections', '0.0')
    
    # Charts
    st.subheader('📊 Detection Analytics')
    
    if not filtered_df.empty:
        # Create charts
        daily_fig, hourly_fig, role_fig = create_detection_charts(filtered_df)
        
        # Display charts in columns
        col1, col2 = st.columns(2)
        
        with col1:
            if daily_fig:
                st.plotly_chart(daily_fig, use_container_width=True)
            if role_fig:
                st.plotly_chart(role_fig, use_container_width=True)
        
        with col2:
            if hourly_fig:
                st.plotly_chart(hourly_fig, use_container_width=True)
        
        # Person activity chart
        st.subheader('👥 Person Activity Analysis')
        person_fig = create_person_activity_chart(filtered_df)
        if person_fig:
            st.plotly_chart(person_fig, use_container_width=True)
        
        # Recent activity table
        st.subheader('🕒 Recent Activity')
        recent_df = filtered_df.head(20).copy()
        recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp'])
        recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(recent_df[['timestamp', 'name', 'role', 'event']], 
                    use_container_width=True, hide_index=True)
        
        # Export functionality
        st.subheader('📤 Export Data')
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f'face_recognition_logs_{start_date}_to_{end_date}.csv',
                mime='text/csv'
            )
        
        with col2:
            # Create a simple Excel-like report
            excel_buffer = io.BytesIO()
            filtered_df.to_excel(excel_buffer, index=False)
            excel_data = excel_buffer.getvalue()
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name=f'face_recognition_report_{start_date}_to_{end_date}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
    
    else:
        st.info('No data found for the selected filters. Try adjusting your search criteria.')

# --- View Logs & Faces ---
else:
    st.header('📊 Event Logs')
    
    # Quick filters for logs
    st.subheader('🔍 Quick Filters')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        log_person_filter = st.text_input('Search by Name', key='log_person', placeholder='Enter name...')
    
    with col2:
        log_role_filter = st.selectbox('Filter by Role', ['All', 'customer', 'employee', 'fraudster'], key='log_role')
    
    with col3:
        log_days = st.selectbox('Time Range', ['All', 'Today', 'Last 7 days', 'Last 30 days'], key='log_days')
    
    # Apply filters to logs
    df = get_db_logs()
    
    if log_person_filter:
        df = df[df['name'].str.contains(log_person_filter, case=False, na=False)]
    
    if log_role_filter != 'All':
        df = df[df['role'] == log_role_filter]
    
    if log_days != 'All':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if log_days == 'Today':
            df = df[df['timestamp'].dt.date == datetime.now().date()]
        elif log_days == 'Last 7 days':
            df = df[df['timestamp'] >= (datetime.now() - timedelta(days=7))]
        elif log_days == 'Last 30 days':
            df = df[df['timestamp'] >= (datetime.now() - timedelta(days=30))]
    
    # Display filtered logs
    st.dataframe(df, use_container_width=True)
    
    # Log statistics
    if not df.empty:
        st.subheader('📈 Log Statistics')
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric('Total Logs', len(df))
        
        with col2:
            st.metric('Unique People', df['name'].nunique())
        
        with col3:
            st.metric('Fraudster Alerts', len(df[df['role'] == 'fraudster']))
        
        with col4:
            if len(df) > 0:
                # Convert timestamp to datetime for display
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                latest_time = df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
                st.metric('Latest Detection', latest_time)
            else:
                st.metric('Latest Detection', 'None')
    
    st.markdown('---')
    st.header('🖼️ Registered Faces')
    
    # Face search
    face_search = st.text_input('Search Registered Faces', placeholder='Enter name...')
    registered_faces = get_registered_faces()
    
    if face_search:
        registered_faces = [face for face in registered_faces 
                          if face_search.lower() in face['name'].lower()]
    
    if registered_faces:
        for face in registered_faces:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(face['img_path'], width=150)
            with col2:
                st.write(f"**Name:** {face['name']}")
                st.write(f"**Role:** {face['role']}")
                st.write(f"**File:** {face['filename']}")
    else:
        st.info('No faces found matching your search criteria.') 