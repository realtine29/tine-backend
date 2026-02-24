import cv2
import numpy as np
import time
import threading
import os
import sys
import json
import cloudinary
import cloudinary.uploader
import cloudinary.api
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import firebase_admin
from firebase_admin import credentials, firestore

from collections import deque
from flask import Flask, Response, request
from flask_cors import CORS
from ultralytics import YOLO
from tensorflow.keras.models import load_model

#  1. CONFIGURATION & CONSTANTS

ENABLE_EMAILS = True 

EMAIL_SENDER = "realinochristine55@gmail.com"
EMAIL_PASSWORD = "wxapomrlleploeqq" 

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

# TIMERS 
HISTORY_SECONDS = 20               
FPS_ESTIMATE = 15
HISTORY_LEN = HISTORY_SECONDS * FPS_ESTIMATE
STILLNESS_LIMIT = 20 * FPS_ESTIMATE 

SCAN_WINDOW_SEC = 4 
SCAN_LEN = SCAN_WINDOW_SEC * FPS_ESTIMATE

LOGIC_SKIP = 2 

ALERT_MAX_DURATION = 10.0           
ROUTINE_COOLDOWN = 20.
DEBUG_MODE = True 


#  VIDEO CONFIGURATION 
TARGET_FPS = 30.0               
FRAME_DELAY = 1.0 / TARGET_FPS 
BUFFER_SECONDS = 60             
BUFFER_SIZE = int(TARGET_FPS * BUFFER_SECONDS)

# Global flag for email limit
EMAIL_LIMIT_REACHED = False


import cv2
import time


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


cameras_dict = {}  # key = camera_name, value = RTSPVideoStream object

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

    # Add camera to memory
    cameras_dict[camera_name] = RTSPVideoStream(rtsp_url, name=camera_name)

    # Save to Firestore
    db.collection("cameras").add({
        "name": camera_name,
        "rtsp_url": rtsp_url,
        "owner": user_id,
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
    def __init__(self, src, name=None):
        self.src = src
        self.name = name
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.ret = False
        self.frame = None
        self.stopped = False
        self.online = self.cap.isOpened()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.ret = True
                    self.frame = frame
                    self.online = True
                else:
                    self.ret = False
                    self.online = False
                    print(f"⚠️ RTSP lost for {self.name} – reconnecting...")
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            else:
                # Try to reconnect if closed
                self.cap.release()
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
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

            if not camera_name or not rtsp_url:
                print("⚠️ Skipping invalid camera document:", data)
                continue

            cameras[camera_name] = RTSPVideoStream(rtsp_url, name=camera_name)
            print(f"✅ Loaded camera: {camera_name}")

        except Exception as e:
            print(f"❌ Failed to load camera: {e}")

    return cameras



cameras_dict = load_cameras_from_firestore()


cloudinary.config(
    cloud_name="dpkds0mpw",
    api_key="573419718341736",
    api_secret="wFzLdsZ_O9vLiZKP2bvZCN_XGjA",
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
def get_all_user_emails():
    emails = []
    try:
        users_ref = db.collection("users").stream()
        for doc in users_ref:
            user_data = doc.to_dict()
            if "email" in user_data: emails.append(user_data["email"])
    except Exception as e: print(f"!! Error fetching emails: {e}")
    return emails

def send_email_alert(label, cloud_url):
    global EMAIL_LIMIT_REACHED
    
    if not ENABLE_EMAILS:
        return

    if EMAIL_LIMIT_REACHED:
        print("!! Skip Email: Daily Limit Reached.")
        return

    recipient_list = get_all_user_emails()
    if not recipient_list: return
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        for email in recipient_list:
            msg = MIMEMultipart()
            msg['From'] = f"Security System <{EMAIL_SENDER}>"
            msg['To'] = email
            msg['Subject'] = f" ALERT: {label} Detected!"
            body = f"Incident: {label}\nView Evidence: {cloud_url}"
            msg.attach(MIMEText(body, 'html'))
            server.send_message(msg)
        server.quit()
        print(f"Email Sent to {len(recipient_list)} users.")
    except Exception as e: 
        print(f"!! SMTP Failed: {e}")
        err_str = str(e)
        if "limit exceeded" in err_str or "5.4.5" in err_str:
            print("!! DAILY EMAIL LIMIT REACHED. PAUSING EMAILS FOR THIS SESSION.")
            EMAIL_LIMIT_REACHED = True

def save_to_firebase(label, cloud_url, accuracy, camera_name):
    try:
        accuracy_decimal = float(accuracy) / 100.0 if accuracy > 0 else 0.0
        local_timestamp = datetime.now()

        db.collection("detections").add({
            "camera_name": camera_name,
            "type": label,
            "video_url": cloud_url,
            "accuracy": accuracy_decimal,
            "timestamp": local_timestamp
        })
    except Exception as e:
        print(f"!! Firebase Error: {e}")


def save_alert_clip(frame_buffer, label, track_id, accuracy=0, fps=30, suspect_count=1, camera_name=""):
   
    try:
        frames = list(frame_buffer)
        if not frames: return

        ts_display = time.strftime("%Y-%m-%d %H:%M:%S")
        ts_short = time.strftime("%H:%M:%S")
        ts_file = time.strftime("%Y%m%d_%H%M%S")
        
        safe_label = label.replace(":", "").replace(" ", "_")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        fn = f"{OUTPUT_DIR}/{camera_name}_{safe_label}_ID{track_id}_{ts_file}.mp4"
        h, w, _ = frames[0].shape
        write_fps = max(fps, 5.0) 
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(fn, fourcc, write_fps, (w, h))
        
        if not out.isOpened():
             print("!! MP4v failed. Using XVID...")
             fn = fn.replace(".mp4", ".avi")
             fourcc = cv2.VideoWriter_fourcc(*'XVID')
             out = cv2.VideoWriter(fn, fourcc, write_fps, (w, h))
        
        if not out.isOpened():
             return

 #DESIGN FOR VIDEO CLIP
        header_h = 40  
        overlay_color = (0, 0, 0) 
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45 
        font_thick = 1

        for f in frames:
            frame_out = f.copy()

            cv2.rectangle(frame_out, (0, h - header_h), (w, h), overlay_color, -1)
            
            header_text = f"{ts_display} | ID:{track_id} {label} | ACC:{accuracy}% | SUSPECTS:{suspect_count}"
            
            (text_w, text_h), baseline = cv2.getTextSize(header_text, font, font_scale, font_thick)
            text_x = 10 
            text_y = h - int((header_h - text_h) / 2) - 5
            
            cv2.putText(frame_out, header_text, (text_x, text_y), font, font_scale, text_color, font_thick, cv2.LINE_AA)
            out.write(frame_out)
            
        out.release()
        
        time.sleep(3.0)
        
        # 2. UPDATE LOCAL JSON
        accuracy_decimal = float(accuracy) / 100.0 if accuracy > 0 else 0.0
        new_entry = {
            "camera_name": camera_name, 
            "timestamp": time.strftime("%B %d, %Y at %I:%M:%S %p"), 
            "type": label, 
            "video_url": fn,
            "accuracy": accuracy_decimal 
        }
        current_logs = []
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, 'r') as f: current_logs = json.load(f)
            except: pass
        current_logs.append(new_entry)
        with open(LOG_FILE, 'w') as f: json.dump(current_logs[-20:], f, indent=2)

        print(f"ALERT SAVED: ID {track_id} - {label} (Suspects: {suspect_count})")
        #Cloudinary
        if os.path.exists(fn) and os.path.getsize(fn) > 0:
            print(f"Uploading ID {track_id}...")
            
            upload_success = False
            attempts = 0
            max_attempts = 5 
            res = None
            
            while attempts < max_attempts and not upload_success:
                try:
                    attempts += 1
                    res = cloudinary.uploader.upload(
                        fn, 
                        resource_type="video", 
                        folder=f"ai_detections/{camera_name}"
                    )
                    upload_success = True
                except Exception as e:
                    print(f"!! Upload Attempt {attempts} Failed: {e}")
                    time.sleep(5)
            
            if upload_success and res:
                raw_url = res.get("secure_url")
                if raw_url:
                    playable_url = raw_url.replace("/upload/", "/upload/vc_h264,f_mp4/")
                    print(f"Link Generated: {playable_url}")
                    
                    save_to_firebase(label, playable_url, accuracy, camera_name) 
                    
                    if ENABLE_EMAILS and 'EMAIL_LIMIT_REACHED' in globals() and not EMAIL_LIMIT_REACHED:
                        if accuracy >= 50: 
                             send_email_alert(label, playable_url)
            else:
                print("!! Upload failed.")
        
    except Exception as e: 
        print(f"!! Error: {e}")
def get_centroid(kpts):
    """Returns the center (x, y) of the person based on keypoints."""
    xs = kpts[0::2]
    ys = kpts[1::2]
    return np.mean(xs), np.mean(ys)

def check_head_scanning(kpts, scan_history):
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
    
    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue
        
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
                        'last_save_time': 0
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
                    if dist_speed > 2: person['loc_hist'].append((cx, cy))
                    if dist_speed < 3.0: person['stationary_counter'] += 1
                    
                    else: person['stationary_counter'] = 0
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
                elif steal_prob > STEAL_THRESH and is_hand_close:
                    raw_label = "CRIME: Stealing"
                    color = (128, 0, 128) 
                    acc = int(steal_prob * 100)

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
                    
                    # PACING LOGIC:
                    # Mataas ang total path (lakad nang lakad) PERO maliit ang displacement (bumabalik sa start).
                    # Binabaan ko ang threshold from 5.0 to 2.5 para mas madali ma-detect.
                    if total_path > (height * 2.5) and net_displacement < (height * 1.0):
                         raw_label = "Pacing"
                         color = (255, 140, 0) 
                         acc = 88
                    
                    # LOITERING (AREA) LOGIC:
                    # Gumagalaw-galaw pero nasa loob lang ng maliit na box (20% ng screen width).
                    elif area_width < (width * 0.2) and area_height < (height * 0.2):
                         raw_label = "Loitering (Area)"
                         color = (0, 255, 255)
                         acc = 92

                # 4. OTHER BEHAVIORS
                elif check_head_scanning(kpts_flat, person['scan_hist']):
                    raw_label = "Suspicious: Scanning"
                    color = (255, 165, 0) 
                    acc = 85
                elif lstm_err > POSE_THRESHOLD:
                    raw_label = "Suspicious Behavior"
                    color = (0, 165, 255)
                    acc = min(int((lstm_err / 0.5) * 100), 99)

                is_validated = validator.get_temporal_validation(track_id, raw_label)
                final_label = raw_label if is_validated else "Normal"
                final_color = color if is_validated else (0, 255, 0)
                
                
                person['current_label'] = final_label
                person['current_color'] = final_color
                person['current_acc'] = acc

                active_suspects = sum(1 for p in people_states.values() if "Stealing" in p.get('current_label', ''))

                if final_label != "Normal":
                    cooldown = 5.0 if "Stealing" in final_label else 30.0
                    if (time.time() - person['last_save_time']) > cooldown:
                         person['last_save_time'] = time.time()
 
                         threading.Thread(target=save_alert_clip, 
                                          args=(list(frame_buffer), final_label, track_id, acc, real_time_fps, active_suspects, camera_name)).start()

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
    return {"cameras": list(cameras_dict.keys())}, 200

@app.route('/video/<camera_name>')
def video_feed(camera_name):
    return Response(gen_frames(camera_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def get_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f: return Response(f.read(), mimetype='application/json')
    return Response("[]", mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)