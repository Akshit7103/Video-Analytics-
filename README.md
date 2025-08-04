# Face Recognition & Logging System (InsightFace/ArcFace)

## Features
- Multi-face, real-time recognition (ArcFace)
- Entry/exit and interaction logging (SQLite)
- Fraudster detection and alerting
- Web dashboard (Streamlit)

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download ArcFace model (InsightFace will auto-download on first run).

## Usage
- Register a new face:
  ```bash
  python register_face.py --name "John Doe" --role customer --image path/to/image.jpg
  ```
- Start real-time recognition and logging:
  ```bash
  python recognize_and_log.py
  ```
- Launch dashboard:
  ```bash
  streamlit run dashboard.py
  ```

## Directory Structure
- `faces/` - Registered face images
- `logs.db` - SQLite database for logs
- `register_face.py` - Add new faces
- `recognize_and_log.py` - Real-time recognition and logging
- `dashboard.py` - Web dashboard

## Notes
- Supports customer, employee, and fraudster roles
- Logs entry/exit and interactions
- Easily extensible for new features 