import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import numpy as np
import time
import threading
import os
import cv2
import sys
import json
import cloudinary
import cloudinary.uploader
import cloudinary.api
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yt_dlp
import firebase_admin
from firebase_admin import credentials, firestore

from collections import deque
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Import security and validation modules from app.py
try:
    from firebase_auth import require_auth, require_role, get_current_user
    from validators import (
        validate_camera_settings,
        validate_alert_filters,
        ValidationError as TINEValidationError
    )
    from rate_limit import (
        init_rate_limiter,
        api_rate_limit,
        detection_rate_limit,
        exempt_from_rate_limit
    )
    from audit_logger import (
        init_audit_logger,
        AuditAction,
        log_audit,
        log_user_action,
        log_error
    )
    from error_handlers import (
        register_error_handlers,
        handle_errors,
        ValidationError,
        error_response
    )
    SECURITY_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Some security modules not available: {e}")
    SECURITY_AVAILABLE = False
    # Define dummy decorators if modules not available
    def require_auth(f): return f
    def require_role(*args): return lambda x: x
    def api_rate_limit(f): return f
    def handle_errors(f): return f
    class ValidationError(Exception): pass

# Import SSE Manager
try:
    from sse_manager import emit_alert, emit_camera_status, emit_detection
    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False
    print("[SSE] SSE manager not available - running without real-time updates")

#  1. CONFIGURATION & CONSTANTS

ENABLE_EMAILS = True 

EMAIL_SENDER = os.environ.get('EMAIL_SENDER', 'realinochristine55@gmail.com')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'bjfy wynj mxtl mjux') 

ANOMALY_MODEL_PATH = 'anomaly_detector.keras'
STEALING_MODEL_PATH = 'stealing_classifier.keras'
YOLO_MODEL = 'yolov8n-pose.pt'

LOG_FILE = "ai_model/detections_log.json"
OUTPUT_DIR = "ai_model/detections"

# THRESHOLDS 
POSE_THRESHOLD = 0.18       
# Aggressive Mode
STEAL_THRESH = 0.20        
# Email Threshold
EMAIL_MIN_ACCURACY = 50 
SCAN_MIN_FRAMES = 5

# TIMERS 
HISTORY_SECONDS = 20               
FPS_ESTIMATE = 15
HISTORY_LEN = HISTORY_SECONDS * FPS_ESTIMATE
STILLNESS_LIMIT = 20 * FPS_ESTIMATE 

SCAN_WINDOW_SEC = 4 
SCAN_LEN = SCAN_WINDOW_SEC * FPS_ESTIMATE

LOGIC_SKIP = 2 

ALERT_MAX_DURATION = 15.0           
ROUTINE_COOLDOWN = 20.
DEBUG_MODE = True 


#  VIDEO CONFIGURATION 
TARGET_FPS = 30.0               
FRAME_DELAY = 1.0 / TARGET_FPS 
BUFFER_SECONDS = 30              
BUFFER_SIZE = int(TARGET_FPS * BUFFER_SECONDS)

# Global flag for email limit
EMAIL_LIMIT_REACHED = False

# Detection tunable settings
DETECTION_DIST_SPEED  = 8.0
DETECTION_PACING_MULT = 1.0
DETECTION_LOITER_W    = 0.35
DETECTION_LOITER_H    = 0.35


#  PRECISION LOGIC HELPERS

class BehaviorValidator:
    def __init__(self):
        self.alert_counters = {} 
        self.MIN_FRAMES_STEAL = 3
        self.MIN_FRAMES_SUSP = 5

    def get_temporal_validation(self, track_id, current_label):
        if track_id not in self.alert_counters:
            self.alert_counters[track_id] = {"label": "Normal", "count": 0}
        
        counter = self.alert_counters[track_id]
        if current_label == counter["label"]:
            counter["count"] += 1
        else:
            counter["label"] = current_label
            counter["count"] = 1

        if "Stealing" in counter["label"] and counter["count"] >= self.MIN_FRAMES_STEAL:
            return True
        if ("Suspicious" in counter["label"] or "Loitering" in counter["label"] or "Pacing" in counter["label"]) and counter["count"] >= self.MIN_FRAMES_SUSP:
            return True
            
        return False

validator = BehaviorValidator()


#  INITIALIZATION
app = Flask(__name__)
CORS(app)

# Initialize security modules if available
if SECURITY_AVAILABLE:
    try:
        init_rate_limiter(app)
        init_audit_logger(app)
        register_error_handlers(app)
        print("[OK] Security modules initialized")
    except Exception as e:
        print(f"[WARNING] Error initializing security modules: {e}")


cameras_dict = {}  # key = camera_name, value = RTSPVideoStream object
@app.route("/add_youtube", methods=["POST"])
def add_youtube():
    data = request.get_json()
    user_id = data.get("userId")
    camera_name = data.get("cameraName")
    youtube_url = data.get("youtubeUrl")

    if not all([user_id, camera_name, youtube_url]):
        return {"error": "Missing data"}, 400

    # Check runtime duplicates in memory
    if camera_name in cameras_dict:
        return {"error": "Camera name already exists"}, 400

    # Permanent Firestore check
    query_name = db.collection("cameras").where("name", "==", camera_name).stream()
    if any(query_name):
        return {"error": "Camera name already exists in database"}, 400

    # 1. Extract the raw stream URL using yt-dlp
    try:
        print(f"[INFO] Extracting stream URL for: {youtube_url}")
        # We prefer a lower resolution like 480p or 720p for faster AI processing
        ydl_opts = {'format': 'best[height<=720]/best', 'quiet': True} 
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            raw_stream_url = info['url']
    except Exception as e:
        print(f"!! Failed to extract YouTube stream: {e}")
        return {"error": "Failed to extract stream from this YouTube link. It might be private or age-restricted."}, 500

    # Kuhanin ang org_id ng user
    org_id = None
    try:
        user_doc = db.collection("users").document(user_id).get()
        if user_doc.exists:
            org_id = user_doc.to_dict().get("org_id", None)
    except: pass

    # 2. Add camera to memory using the RAW stream URL
    cameras_dict[camera_name] = RTSPVideoStream(raw_stream_url, original_url=youtube_url, name=camera_name, is_youtube=True)

    # 3. Save to Firestore (We save the original YouTube URL, not the raw one)
    db.collection("cameras").add({
        "name":       camera_name,
        "rtsp_url":   youtube_url,
        "is_youtube": True,
        "owner":      user_id,
        "org_id":     org_id,
        "created_at": firestore.SERVER_TIMESTAMP
    })

    print(f"[INFO] YouTube Camera added: {camera_name}")
    return {"message": f"YouTube stream '{camera_name}' added successfully!"}, 200

@app.route("/addCamera", methods=["POST"])
def add_camera():
    data = request.get_json()
    user_id = data.get("userId")
    camera_name = data.get("cameraName")
    rtsp_url = data.get("rtspUrl")

    if not all([user_id, camera_name, rtsp_url]):
        return {"error": "Missing data"}, 400

    # Check runtime duplicates in memory
    if camera_name in cameras_dict:
        return {"error": "Camera name already exists"}, 400

    for cam in cameras_dict.values():
        if cam.src == rtsp_url:
            return {"error": "This RTSP URL is already added"}, 400

    # Permanent Firestore check for both camera name and RTSP
    query_name = db.collection("cameras").where("name", "==", camera_name).stream()
    if any(query_name):
        return {"error": "Camera name already exists in database"}, 400

    query_rtsp = db.collection("cameras").where("rtsp_url", "==", rtsp_url).stream()
    if any(query_rtsp):
        return {"error": "This RTSP URL already exists in database"}, 400

    # Kuhanin ang org_id ng user
    org_id = None
    try:
        user_doc = db.collection("users").document(user_id).get()
        if user_doc.exists:
            org_id = user_doc.to_dict().get("org_id", None)
    except: pass

    # Add camera to memory
    cameras_dict[camera_name] = RTSPVideoStream(rtsp_url, name=camera_name)

    # Save to Firestore
    db.collection("cameras").add({
        "name":       camera_name,
        "rtsp_url":   rtsp_url,
        "owner":      user_id,
        "org_id":     org_id,
        "created_at": firestore.SERVER_TIMESTAMP
    })

    print(f"[INFO] Camera added: {camera_name} ({rtsp_url}) by {user_id}")
    return {"message": f"Camera '{camera_name}' added successfully!"}, 200



if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()


# VIDEO STREAM CLASS


class RTSPVideoStream:
    # We added an 'original_url' and 'is_youtube' parameter here
    def __init__(self, src, original_url=None, name=None, is_youtube=False):
        self.src = src
        # Kailangan i-save ang original link para may magamit ang get_fresh_url
        self.original_url = original_url if original_url else src 
        self.name = name
        self.is_youtube = is_youtube
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.ret = False
        self.frame = None
        self.stopped = False
        self.online = self.cap.isOpened()
        
        # --- NEW: Get video FPS to pace the playback ---
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_delay = 1.0 / self.fps
        if not self.fps or self.fps <= 0:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                    # Fix: Kapag nagbigay ng corrupted/extreme FPS ang YouTube, ibalik sa 30fps
            if not self.fps or self.fps <= 0 or self.fps > 120:
                    self.fps = 30.0
        # -----------------------------------------------
        
        threading.Thread(target=self.update, daemon=True).start()

    def get_fresh_url(self):
        print(f"[INFO] Fetching a fresh token for YouTube stream: {self.name}...")
        try:
            import yt_dlp
            ydl_opts = {'format': 'best[height<=720]/best', 'quiet': True, 'extract_flat': False}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.original_url, download=False)
                return info.get('url')
        except Exception as e:
            print(f"!! Failed to refresh URL: {e}")
            return self.src

    def update(self):
        while not self.stopped:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.ret = True
                    self.frame = frame
                    self.online = True
                    if self.is_youtube:
                        time.sleep(self.frame_delay)
                else:
                    self.ret = False
                    self.online = False
                    print(f"⚠️ Stream lost for {self.name} – reconnecting...")
                    self.cap.release()
                    time.sleep(2)
                    
                    # Kukuha muna ng bagong link kung YouTube ito bago mag-reconnect
                    reconnect_url = self.get_fresh_url() if self.is_youtube else self.src
                    self.cap = cv2.VideoCapture(reconnect_url, cv2.CAP_FFMPEG)
            else:
                self.cap.release()
                time.sleep(2)
                reconnect_url = self.get_fresh_url() if self.is_youtube else self.src
                self.cap = cv2.VideoCapture(reconnect_url, cv2.CAP_FFMPEG)
                self.online = self.cap.isOpened()
                if self.online:
                    print(f"✅ Camera {self.name} reconnected!")

    def read(self):
        if not self.cap.isOpened():
            self.online = False
            return False, None
        return self.ret, self.frame
# --- Load cameras from Firestore ---


def load_cameras_from_firestore():
    cams = db.collection("cameras").stream()
    cameras = {}

    for cam_doc in cams:
        data = cam_doc.to_dict()

        try:
            camera_name = data.get("name")        
            rtsp_url = data.get("rtsp_url")
            # Check if this camera was flagged as a YouTube stream when it was added
            is_youtube = data.get("is_youtube", False) 

            if not camera_name or not rtsp_url:
                print("⚠️ Skipping invalid camera document:", data)
                continue

            if is_youtube:
                # It's a YouTube link, so we need to extract a FRESH raw stream URL
                print(f"[INFO] Fetching fresh YouTube stream for: {camera_name}...")
                try:
                    ydl_opts = {'format': 'best[height<=720]/best', 'quiet': True}
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(rtsp_url, download=False)
                        fresh_stream_url = info['url']
                        
                        cameras[camera_name] = RTSPVideoStream(fresh_stream_url, name=camera_name, is_youtube=True)
                        print(f"✅ Loaded YouTube camera: {camera_name}")
                except Exception as e:
                    print(f"❌ Failed to refresh YouTube camera '{camera_name}'. The video might be offline: {e}")
            else:
                # It's a normal RTSP camera, load it directly
                cameras[camera_name] = RTSPVideoStream(rtsp_url, name=camera_name)
                print(f"✅ Loaded RTSP camera: {camera_name}")

        except Exception as e:
            print(f"❌ Failed to load camera: {e}")

    return cameras



cameras_dict = load_cameras_from_firestore()


cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME', 'dpkds0mpw'),
    api_key=os.environ.get('CLOUDINARY_API_KEY', '573419718341736'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET', 'wFzLdsZ_O9vLiZKP2bvZCN_XGjA'),
    secure=True
)

print(">> Loading AI Models on Nitro V 15...")
try:
    lstm_model = load_model(ANOMALY_MODEL_PATH)
    yolo_model = YOLO(YOLO_MODEL)
    stealing_model = load_model(STEALING_MODEL_PATH) if os.path.exists(STEALING_MODEL_PATH) else None
    print(">> Models loaded successfully.")
except Exception as e:
    print(f"!! Error: {e}")
    sys.exit(1)





# HELPER FUNCTIONS
def get_user_emails_by_org(org_id):
    """Kukunin lang ang emails ng STANDARD USERS (hindi admin) na nasa parehong org_id."""
    emails = []
    try:
        if org_id:
            users_ref = db.collection("users").where("role", "==", "user").where("org_id", "==", org_id).stream()
        else:
            users_ref = db.collection("users").where("role", "==", "user").stream()
        
        for doc in users_ref:
            user_data = doc.to_dict()
            if "email" in user_data:
                emails.append(user_data["email"])
    except Exception as e: 
        print(f"!! Error fetching emails: {e}")
    return emails

def send_email_alert(label, cloud_url, camera_name):
    global EMAIL_LIMIT_REACHED
    if not ENABLE_EMAILS or EMAIL_LIMIT_REACHED: return
    
    org_id = get_org_id_for_camera(camera_name)
    recipient_list = get_user_emails_by_org(org_id)
    
    if not recipient_list: 
        print(f"[EMAIL] No standard users found for Org ID: {org_id}. Skipping email.")
        return
        
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        for email in recipient_list:
            msg = MIMEMultipart()
            msg['From'] = f"Security System <{EMAIL_SENDER}>"
            msg['To'] = email
            msg['Subject'] = f" ALERT: {label} Detected!"
            body = f"Incident: {label}<br>Camera: {camera_name}<br>View Evidence: <a href='{cloud_url}'>Click Here</a>"
            msg.attach(MIMEText(body, 'html'))
            server.send_message(msg)
        server.quit()
        print(f"[EMAIL] Sent successfully to {len(recipient_list)} user(s).")
    except Exception as e: 
        print(f"!! SMTP Failed: {e}")
        if "limit" in str(e).lower() or "5.4.5" in str(e):
            EMAIL_LIMIT_REACHED = True

def get_org_id_for_camera(camera_name):
    """Kukunin ang org_id ng camera mula sa Firestore."""
    try:
        # Check cameras collection first
        docs = db.collection("cameras").where("name", "==", camera_name).limit(1).stream()
        for doc in docs:
            data = doc.to_dict()
            # Try to get org_id directly from camera doc first
            if "org_id" in data and data["org_id"]:
                return data["org_id"]
            
            # Fallback to owner's org_id if not directly on camera
            owner_uid = data.get("owner")
            if owner_uid:
                user_doc = db.collection("users").document(owner_uid).get()
                if user_doc.exists:
                    return user_doc.to_dict().get("org_id", "default")
    except Exception as e:
        print(f"!! Error getting org_id for {camera_name}: {e}")
    return "default"


def save_to_firebase(label, cloud_url, confidence_score, camera_name):
    try:
                MODEL_TRAINING_ACCURACY = 0.89 
                
                # 2. REAL-TIME CONFIDENCE (Yung pabago-bagong hinala ng AI)
                conf_pct = int(confidence_score)
                if conf_pct >= 80:
                    conf_str = f"{conf_pct}% (HIGH)"
                elif conf_pct >= 50:
                    conf_str = f"{conf_pct}% (MODERATE)"
                else:
                    conf_str = f"{conf_pct}% (LOW)"

                local_timestamp  = datetime.now()
                org_id           = get_org_id_for_camera(camera_name)

                db.collection("detections").add({
                    "camera_name": camera_name,
                    "type":        label,
                    "video_url":   cloud_url,
                    "accuracy":    MODEL_TRAINING_ACCURACY, 
                    "confidence":  conf_str,
                    "timestamp":   local_timestamp,
                    "created_at":  firestore.SERVER_TIMESTAMP,
                    "org_id":      org_id,
                    "duration":    f"{ALERT_MAX_DURATION}s"
                })
                print(f"[FIREBASE] Alert log sent successfully for {camera_name} ({label})")
    except Exception as e:
        print(f"!! Firebase Error: {e}")


def save_alert_clip(frame_buffer, label, track_id, confidence_score, fps=30, suspect_count=1, camera_name=""):
    try:
        # Calculate how many frames represent the exact duration
        max_frames = int(fps * ALERT_MAX_DURATION)
        frames = list(frame_buffer)[-max_frames:]
        if not frames: return

        ts_display = time.strftime("%Y-%m-%d %H:%M:%S")
        ts_file = time.strftime("%Y%m%d_%H%M%S")
        
        safe_label = label.replace(":", "").replace(" ", "_")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        fn = f"{OUTPUT_DIR}/{camera_name}_{safe_label}_ID{track_id}_{ts_file}.mp4"
        h, w, _ = frames[0].shape
        write_fps = max(fps, 5.0) 
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(fn, fourcc, write_fps, (w, h))
        
        if not out.isOpened():
             fn = fn.replace(".mp4", ".avi")
             fourcc = cv2.VideoWriter_fourcc(*'XVID')
             out = cv2.VideoWriter(fn, fourcc, write_fps, (w, h))
        if not out.isOpened(): return

        header_h = 40  
        overlay_color = (0, 0, 0) 
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45 
        font_thick = 1

        for f in frames:
            frame_out = f.copy()
            cv2.rectangle(frame_out, (0, h - header_h), (w, h), overlay_color, -1)
            header_text = f"{ts_display} | ID:{track_id} {label} | CONF:{confidence_score}% | SUSPECTS:{suspect_count}"
            (text_w, text_h), baseline = cv2.getTextSize(header_text, font, font_scale, font_thick)
            cv2.putText(frame_out, header_text, (10, h - int((header_h - text_h) / 2) - 5), font, font_scale, text_color, font_thick, cv2.LINE_AA)
            out.write(frame_out)
            
        out.release()
        time.sleep(1.0) # Binawasan natin ang sleep para mas mabilis
        
        # 1. Cloudinary Upload
        playable_url = None
        if os.path.exists(fn) and os.path.getsize(fn) > 0:
            print(f"[UPLOAD] Uploading ID {track_id} video to Cloudinary...")
            upload_success = False
            attempts = 0
            res = None
            while attempts < 3 and not upload_success:
                try:
                    attempts += 1
                    res = cloudinary.uploader.upload(fn, resource_type="video", folder=f"ai_detections/{camera_name}")
                    upload_success = True
                except Exception as e: 
                    print(f"Cloudinary attempt {attempts} failed: {e}")
                    time.sleep(3)
                    
            if upload_success and res:
                raw_url = res.get("secure_url")
                if raw_url:
                    playable_url = raw_url.replace("/upload/", "/upload/vc_h264,f_mp4/")
                    print(f"[UPLOAD] Success: {playable_url}")
                    
                    # Notify the frontend via SSE that the video is ready
                    if SSE_AVAILABLE:
                        emit_alert({
                            "camera_name": camera_name,
                            "type": label,
                            "video_url": playable_url,
                            "confidence": f"{confidence_score}%",
                            "track_id": track_id,
                            "suspects": suspect_count,
                            "timestamp": ts_display,
                            "org_id": get_org_id_for_camera(camera_name)
                        })
                    
        # 2. Save to Firebase (ITO YUNG PUMAPASOK SA DASHBOARD LOGS)
        save_to_firebase(label, playable_url, confidence_score, camera_name) 
                    
        # 3. Send Email (Para sa Standard Users lang)
        if playable_url and ENABLE_EMAILS and 'EMAIL_LIMIT_REACHED' in globals() and not EMAIL_LIMIT_REACHED:
            if confidence_score >= 50: 
                send_email_alert(label, playable_url, camera_name)
                            
    except Exception as e: 
        print(f"!! Critical Error in save_alert_clip: {e}")


def get_centroid(kpts):
    """Returns the center (x, y) of the person based on keypoints."""
    xs = kpts[0::2]
    ys = kpts[1::2]
    return np.mean(xs), np.mean(ys)

def check_head_scanning(kpts, scan_history):
    global SCAN_MIN_FRAMES
    """Detects sustained scanning (looking left then right)."""
    nose_x = kpts[0]
    left_sh_x = kpts[10] 
    right_sh_x = kpts[12] 
    shoulder_width = abs(right_sh_x - left_sh_x)
   
    if shoulder_width < 0.01: return False
   
    look_ratio = (nose_x - left_sh_x) / shoulder_width

    if look_ratio < 0.15: scan_history.append("LEFT")
    elif look_ratio > 0.85: scan_history.append("RIGHT")
    else: scan_history.append("CENTER")

    return scan_history.count("LEFT") > 5 and scan_history.count("RIGHT") > 5

def is_hand_near_face(kpts, height_px):
    nose_x, nose_y = kpts[0], kpts[1]
    lw_x, lw_y = kpts[18], kpts[19]
    rw_x, rw_y = kpts[20], kpts[21]
   
    dist_l = np.hypot(lw_x - nose_x, lw_y - nose_y) * height_px
    dist_r = np.hypot(rw_x - nose_x, rw_y - nose_y) * height_px
   
    return dist_l < (height_px * 0.15) or dist_r < (height_px * 0.15)

def is_hand_in_stashing_zone(kpts, h_px):
    """Checks if hands are near hips or pockets relative to torso size."""
    lw_x, lw_y = kpts[18], kpts[19]
    rw_x, rw_y = kpts[20], kpts[21]
    l_hip_x, l_hip_y = kpts[22], kpts[23]
    r_hip_x, r_hip_y = kpts[24], kpts[25]
   
    shoulder_y = (kpts[11] + kpts[13]) / 2 
    torso_h = abs((l_hip_y + r_hip_y)/2 - shoulder_y)
    
    # 55% of torso height is a reliable stashing zone
    dynamic_thresh = max(torso_h * 0.55, 0.05) 
   
    dist_l = np.hypot(lw_x - l_hip_x, lw_y - l_hip_y)
    dist_r = np.hypot(rw_x - r_hip_x, rw_y - r_hip_y)
   
    return min(dist_l, dist_r) < dynamic_thresh

# MAIN LOGIC LOOP
people_states = {}

def gen_frames(camera_name):
    camera = cameras_dict.get(camera_name)
    if not camera:
        print(f"Camera '{camera_name}' not found.")
        return
    
    frame_buffer = deque(maxlen=BUFFER_SIZE)
    frame_count = 0
    LOGIC_SKIP = 2 
    
    prev_frame_time = 0
    new_frame_time = 0
    real_time_fps = 15.0 
    last_processed_frame = None
    
    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue
        if frame is last_processed_frame:
            time.sleep(0.01) 
            continue
        last_processed_frame = frame
        
        new_frame_time = time.time()
        if prev_frame_time > 0:
            fps_instant = 1 / (new_frame_time - prev_frame_time)
            real_time_fps = 0.9 * real_time_fps + 0.1 * fps_instant
        prev_frame_time = new_frame_time

        orig_h, orig_w = frame.shape[:2]
        target_w = 640
        target_h = int(orig_h * (target_w / orig_w))
        frame = cv2.resize(frame, (target_w, target_h))
        height, width = frame.shape[:2]
        
        results = yolo_model.track(frame, persist=True, verbose=False, classes=[0], conf=0.45)
       
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.xyn.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
                
            for box, track_id, kpts, y_conf in zip(boxes, track_ids, keypoints, confidences):
                kpts_flat = kpts.flatten()
                
                if track_id not in people_states:
                    people_states[track_id] = {
                        'pose_seq': [], 
                        'loc_hist': deque(maxlen=HISTORY_LEN),
                        'scan_hist': deque(maxlen=SCAN_LEN),
                        'stationary_counter': 0,
                        'alert_start_time': 0,
                        'current_label': "Normal", 'current_color': (0, 255, 0),
                        'smoothed_box': box, 'current_acc': 0,
                        'last_lstm_err': 0.0, 'last_steal_prob': 0.0,
                        'last_save_time': 0,
                        'alert_saved': False,
                        'alert_pending_save': False,
                        'save_scheduled_time': 0,
                        'alert_data': None
                    }

                
                person = people_states[track_id]
                alpha = 0.5 
                person['smoothed_box'] = (alpha * box) + ((1.0 - alpha) * person['smoothed_box'])
                
                person['pose_seq'].append(kpts_flat)
                if len(person['pose_seq']) > 30: person['pose_seq'].pop(0)

                use_box = person['smoothed_box']
                x1, y1, x2, y2 = map(int, use_box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                dist_speed = 0.0
                if len(person['loc_hist']) > 0:
                    lx, ly = person['loc_hist'][-1]
                    dist_speed = np.hypot(cx-lx, cy-ly)
                    if dist_speed > 2:
                        person['loc_hist'].append((cx, cy))
                    if dist_speed < DETECTION_DIST_SPEED:
                        person['stationary_counter'] += 1
                    else:
                        person['stationary_counter'] = 0
                else:
                    person['loc_hist'].append((cx, cy))

                # AI Logic
                if len(person['pose_seq']) == 30 and (frame_count % LOGIC_SKIP == 0):
                    inp = np.array([person['pose_seq']])
                    lstm_out = lstm_model.predict(inp, verbose=0)
                    lstm_err = np.mean(np.abs(lstm_out - inp))
                    steal_prob = stealing_model.predict(inp, verbose=0)[0][0] if stealing_model else 0
                    
                    person['last_lstm_err'] = lstm_err
                    person['last_steal_prob'] = steal_prob
                else:
                    lstm_err = person.get('last_lstm_err', 0.0)
                    steal_prob = person.get('last_steal_prob', 0.0)

                raw_label = "Normal"
                color = (0, 255, 0)
                acc = int(y_conf * 100)

                is_hand_close = is_hand_in_stashing_zone(kpts_flat, height)
                
                #  BEHAVIOR LOGIC           
                # 1. PRIORITY: LOITERING (Stillness)
                # Uunahin natin ito. Kung hindi gumagalaw ang tao, LOITERING 'yan.
                # Hindi dapat ito ma-override ng 'Stealing' glitch.
                if person['stationary_counter'] > STILLNESS_LIMIT:
                    raw_label = "Loitering (Still)"
                    color = (0, 255, 255) 
                    acc = 90

                # 2. PRIORITY: STEALING (Sensitive Hand Zone + High Threshold)
                # Ilalagay natin ito sa pangalawa. At dadagdagan natin ng 'Net Displacement' check.
                # Kung ang tao ay gumagalaw nang mabilis (walking), mahirap masabing stealing 'yun agad.
                # Dapat ay 'focused' siya o kaya ay mabagal ang galaw habang nagnanakaw.
# 2. PRIORITY: STEALING 
                elif steal_prob > STEAL_THRESH:
                        raw_label = "Anomaly: Stealing"
                        color = (128, 0, 128) 
                        
                        # ACADEMIC PROBABILITY CALIBRATION (Min-Max Scaling)
                        # I-mamap natin ang 0.20 -> 50% (Baseline ng detection)
                        # At ang 1.0 -> 99%
                        scaled_acc = 50.0 + ((steal_prob - STEAL_THRESH) / (1.0 - STEAL_THRESH)) * (99.0 - 50.0)
                        
                        # Stashing Zone Bonus (+15%)
                        bonus_acc = 15 if is_hand_close else 0
                        
                        acc = min(int(scaled_acc + bonus_acc), 99)

                # 3. PRIORITY: MOTION (Pacing & Area Loitering)
                # Binabaan natin ang requirement: kahit kalahati pa lang ng history (10 secs), pwede na mag-detect.
                elif len(person['loc_hist']) >= (HISTORY_LEN // 2):
                    xs = [p[0] for p in person['loc_hist']]
                    ys = [p[1] for p in person['loc_hist']]
                    
                    # Calculate Path (Total nilakad) vs Displacement (Layo sa start point)
                    total_path = sum(np.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1]) for i in range(1, len(xs)))
                    net_displacement = np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
                    area_width = max(xs) - min(xs)
                    area_height = max(ys) - min(ys)

                    # PACING LOGIC
                    if total_path > (height * DETECTION_PACING_MULT) and net_displacement < (total_path * 0.5):
                         raw_label = "Anomaly:Pacing"
                         color = (255, 140, 0)
                         acc = 88

                    # LOITERING (AREA) LOGIC
                    elif area_width < (width * DETECTION_LOITER_W) and area_height < (height * DETECTION_LOITER_H):
                         raw_label = "Anomaly:Loitering (Area)"
                         color = (0, 255, 255)
                         acc = 92

                # 4. OTHER BEHAVIORS
                elif check_head_scanning(kpts_flat, person['scan_hist']):
                    raw_label = "Anomaly: Scanning"
                    color = (255, 165, 0) 
                    acc = 85
                is_validated = validator.get_temporal_validation(track_id, raw_label)
                final_label = raw_label if is_validated else "Normal"
                final_color = color if is_validated else (0, 255, 0)
                
                
                person['current_label'] = final_label
                person['current_color'] = final_color
                person['current_acc'] = acc

                active_suspects = sum(1 for p in people_states.values() if "Stealing" in p.get('current_label', ''))
                
                if final_label != "Normal" and not person.get('alert_saved', False) and not person.get('alert_pending_save', False):
                    cooldown = 60.0 if "Stealing" in final_label else 30.0
                    if (time.time() - person.get('last_save_time', 0)) > cooldown:
                        person['alert_pending_save'] = True
                        person['save_scheduled_time'] = time.time() + 7.5 # Wait 7.5s for post-roll
                        person['alert_data'] = (final_label, track_id, acc, active_suspects)
                        
                        # Emit immediate detection event via SSE
                        if SSE_AVAILABLE:
                            emit_detection(camera_name, final_label, acc)

                if person.get('alert_pending_save', False) and time.time() >= person['save_scheduled_time']:
                    person['alert_pending_save'] = False
                    person['alert_saved'] = True
                    person['last_save_time'] = time.time()
                    p_label, p_track_id, p_acc, p_suspects = person['alert_data']
                    
                    # Passing a copy of the current frame_buffer (last 15 seconds)
                    threading.Thread(target=save_alert_clip, 
                                      args=(list(frame_buffer), p_label, p_track_id, p_acc, real_time_fps, p_suspects, camera_name)).start()

                if final_label == "Normal":
                    person['alert_saved'] = False
                    person['alert_pending_save'] = False # Cancel pending if it returns to normal too quickly? 
                    # Actually, better to keep it if we want to see what happened. 
                    # But the current logic resets alert_saved on Normal.
                    # Let's keep it consistent.
                            
                bx1, by1, bx2, by2 = map(int, person['smoothed_box'])
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), person['current_color'], 2)
                
                label_text = f"{person['current_label']} ({person['current_acc']}%)"
                cv2.putText(frame, f"ID {track_id}", (bx1, by1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, person['current_color'], 2)
                cv2.putText(frame, label_text, (bx1, by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, person['current_color'], 2)
                
                for i in range(0, 17):
                    xk, yk = int(kpts_flat[i*2] * width), int(kpts_flat[i*2+1] * height)
                    if xk > 0 and yk > 0: cv2.circle(frame, (xk, yk), 4, (0, 255, 0), -1)

        frame_count += 1
        frame_buffer.append(frame.copy())

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


#  FLASK ROUTES
@app.route('/cameras', methods=['GET'])
def get_cameras():
    org_id = request.args.get("org_id", None)
    cam_list = []

    # Kunin lahat ng cameras mula sa Firestore
    try:
        if org_id:
            # I-filter ang cameras sa Firestore base sa org_id
            docs = db.collection("cameras").where("org_id", "==", org_id).stream()
        else:
            # Walang org_id — ibalik lahat (para sa superadmin)
            docs = db.collection("cameras").stream()
        
        firestore_cams = {doc.to_dict().get("name"): doc.to_dict() for doc in docs}
    except Exception as e:
        print(f"!! Firestore query error: {e}")
        firestore_cams = {}

    # I-check din ang memory para sa online status
    for name, firestore_data in firestore_cams.items():
        # Get online status from memory if available
        online = False
        cam_type = "rtsp"
        src = firestore_data.get("rtsp_url", "")
        
        if name in cameras_dict:
            cam = cameras_dict[name]
            online = getattr(cam, 'online', False)
            cam_type = "youtube" if getattr(cam, 'is_youtube', False) else "rtsp"
            src = getattr(cam, 'original_url', getattr(cam, 'src', src))
        
        cam_list.append({
            "name":   name,
            "online": online,
            "type":   cam_type,
            "src":    src,
            "org_id": firestore_data.get("org_id", "default")
        })

    return {"cameras": cam_list}, 200


@app.route('/delete_camera/<camera_name>', methods=['DELETE'])
def delete_camera(camera_name):
    if camera_name not in cameras_dict:
        return {"error": "Camera not found"}, 404

    # I-stop ang camera stream
    cam = cameras_dict[camera_name]
    if hasattr(cam, 'stop'):
        cam.stop()
    elif hasattr(cam, 'stopped'):
        cam.stopped = True
    del cameras_dict[camera_name]

    # I-delete sa Firestore
    try:
        docs = db.collection("cameras").where("name", "==", camera_name).stream()
        for doc in docs:
            doc.reference.delete()
        print(f"[INFO] Camera deleted: {camera_name}")
    except Exception as e:
        print(f"!! Firestore delete error: {e}")

    return {"message": f"Camera '{camera_name}' deleted successfully!"}, 200
@app.route('/video/<camera_name>')
def video_feed(camera_name):
    return Response(gen_frames(camera_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream')
@exempt_from_rate_limit
def stream():
    """
    SSE stream endpoint. If sse_manager is available, it uses the proper
    SSE response creator to allow for persistent real-time updates.
    """
    if SSE_AVAILABLE:
        try:
            from sse_manager import create_sse_response
            events_param = request.args.get('events', 'alert,camera_status,detection,health')
            event_types = [e.strip() for e in events_param.split(',')]
            client_id = request.args.get('client_id', f"ai-{time.time()}")
            return create_sse_response(event_types, client_id)
        except Exception as e:
            print(f"[SSE] Error creating SSE response: {e}")

    # Fallback/Heartbeat for when sse_manager is not fully available
    def generate():
        while True:
            yield "data: {\"type\": \"heartbeat\"}\n\n"
            time.sleep(30)
    return Response(generate(), mimetype="text/event-stream")

@app.route('/logs', methods=['GET'])
@exempt_from_rate_limit
def get_logs():
    org_id = request.args.get("org_id", None)
    if not org_id:
        return jsonify([])
    
    try:
        # NOTE: If this fails, check your Firebase Console for a "Composite Index" requirement link.
        query = db.collection("detections").where("org_id", "==", org_id)
        
        # Try ordering by timestamp - if this fails without an index, it will catch below
        docs = query.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(20).stream()
        
        logs = []
        for doc in docs:
            d = doc.to_dict()
            if 'timestamp' in d and hasattr(d['timestamp'], 'strftime'):
                d['timestamp'] = d['timestamp'].strftime("%B %d, %Y at %I:%M:%S %p")
            d['id'] = doc.id 
            logs.append(d)
            
        return jsonify(logs)
    except Exception as e:
        print(f"!! Firebase Logs Fetch Error: {e}")
        # Fallback: Try fetching without ordering if index is missing (to at least show something)
        try:
            docs = db.collection("detections").where("org_id", "==", org_id).limit(20).stream()
            logs = []
            for doc in docs:
                d = doc.to_dict()
                if 'timestamp' in d and hasattr(d['timestamp'], 'strftime'):
                    d['timestamp'] = d['timestamp'].strftime("%B %d, %Y at %I:%M:%S %p")
                d['id'] = doc.id
                logs.append(d)
            return jsonify(logs)
        except Exception as e2:
            print(f"!! Fallback Fetch Error: {e2}")
            return jsonify([])


@app.route('/detection_settings', methods=['GET'])
def get_detection_settings():
    return {
        "stillness_limit_seconds": STILLNESS_LIMIT / FPS_ESTIMATE,
        "dist_speed_threshold":    DETECTION_DIST_SPEED,
        "history_seconds":         HISTORY_SECONDS,
        "pacing_path_mult":        DETECTION_PACING_MULT,
        "loiter_area_w":           DETECTION_LOITER_W,
        "loiter_area_h":           DETECTION_LOITER_H,
        "pose_threshold":          POSE_THRESHOLD, 
        "steal_threshold":         STEAL_THRESH,
        "scan_threshold":          SCAN_MIN_FRAMES,
        "video_fps":               TARGET_FPS # Bagong control para sa mabilis na videos
    }, 200


@app.route('/detection_settings', methods=['POST'])
def update_detection_settings():
    global STILLNESS_LIMIT, HISTORY_SECONDS, HISTORY_LEN
    global STEAL_THRESH, SCAN_MIN_FRAMES, POSE_THRESHOLD
    global DETECTION_DIST_SPEED, DETECTION_PACING_MULT
    global DETECTION_LOITER_W, DETECTION_LOITER_H, TARGET_FPS

    data = request.get_json()

    if "stillness_limit_seconds" in data:
        STILLNESS_LIMIT = float(data["stillness_limit_seconds"]) * FPS_ESTIMATE
    if "history_seconds" in data:
        HISTORY_SECONDS = float(data["history_seconds"])
        HISTORY_LEN     = int(HISTORY_SECONDS * FPS_ESTIMATE)
    if "steal_threshold" in data:
        STEAL_THRESH = float(data["steal_threshold"])
    if "pose_threshold" in data:
        POSE_THRESHOLD = float(data["pose_threshold"])
    if "scan_threshold" in data:
        SCAN_MIN_FRAMES = int(data["scan_threshold"])
    if "dist_speed_threshold" in data:
        DETECTION_DIST_SPEED = float(data["dist_speed_threshold"])
    if "pacing_path_mult" in data:
        DETECTION_PACING_MULT = float(data["pacing_path_mult"])
    if "loiter_area_w" in data:
        DETECTION_LOITER_W = float(data["loiter_area_w"])
    if "loiter_area_h" in data:
        DETECTION_LOITER_H = float(data["loiter_area_h"])
    if "video_fps" in data:
        TARGET_FPS = float(data["video_fps"])

    print(f"[INFO] Detection settings updated: {data}")
    return {"message": "Settings updated successfully!"}, 200

@app.route('/delete_user/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        from firebase_admin import auth as firebase_auth
        firebase_auth.delete_user(user_id)
        print(f"[INFO] User deleted from Auth: {user_id}")
        return {"message": f"User {user_id} deleted from Firebase Auth."}, 200
    except Exception as e:
        print(f"!! Error deleting user from Auth: {e}")
        return {"error": str(e)}, 500

from flask import send_from_directory
import os

# PALITAN ITO NG TOTOONG DRIVE LETTER NG SD CARD MO (e.g., 'E:/CCTV_Records' o 'D:/Records')
SD_CARD_PATH = "D:/CCTV_Records" 

@app.route('/get_recordings', methods=['GET'])
def get_recordings():
    camera_name = request.args.get("camera")
    date_str = request.args.get("date") # Format: YYYY-MM-DD
    
    if not camera_name or not date_str:
        return {"error": "Missing parameters"}, 400
        
    # Expected folder structure: D:/CCTV_Records/Living_Room_Camera/2026-03-23/
    folder_path = os.path.join(SD_CARD_PATH, camera_name, date_str)
    
    if not os.path.exists(folder_path):
        return {"files": []}, 200
        
    # Kuhanin lahat ng .mp4 files sa loob ng folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    files.sort() # I-sort para sunod-sunod ang oras
    
    return {"files": files}, 200

@app.route('/play_record', methods=['GET'])
def play_record():
    camera_name = request.args.get("camera")
    date_str = request.args.get("date")
    file_name = request.args.get("file")
    
    folder_path = os.path.join(SD_CARD_PATH, camera_name, date_str)
    
    if not os.path.exists(os.path.join(folder_path, file_name)):
        return "Video not found", 404
        
    # Ise-serve ng Flask ang video file papunta sa React
    return send_from_directory(folder_path, file_name, mimetype='video/mp4')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)