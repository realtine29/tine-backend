import cv2
import numpy as np
import time
import threading
import os
import sys
import json
import subprocess
import cloudinary
import cloudinary.uploader
import cloudinary.api
import smtplib
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import firebase_admin
from firebase_admin import credentials, firestore

from collections import deque
from flask import Flask, Response, request, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# ============================================================
#  1. CONFIGURATION & CONSTANTS
# ============================================================

ENABLE_EMAILS = True

EMAIL_SENDER   = "realinochristine55@gmail.com"
EMAIL_PASSWORD = "llkg rmhj bbno xigl"

ANOMALY_MODEL_PATH  = 'anomaly_detector.keras'
STEALING_MODEL_PATH = 'stealing_classifier.keras'
YOLO_MODEL          = 'yolov8n-pose.pt'

LOG_FILE   = "ai_model/detections_log.json"
OUTPUT_DIR = "ai_model/detections"

HLS_DIR = "hls_output"
os.makedirs(HLS_DIR, exist_ok=True)

POSE_THRESHOLD     = 0.22   # hindi binago
STEAL_THRESH       = 0.55   # hindi binago
EMAIL_MIN_ACCURACY = 50

ACC_HIGH   = 85
ACC_MEDIUM = 65

FPS_ESTIMATE    = 15
HISTORY_SECONDS = 15
HISTORY_LEN     = HISTORY_SECONDS * FPS_ESTIMATE
SCAN_WINDOW_SEC = 8
SCAN_LEN        = SCAN_WINDOW_SEC * FPS_ESTIMATE
STILLNESS_LIMIT = 15 * FPS_ESTIMATE
LOGIC_SKIP      = 2

TARGET_FPS     = 30.0
BUFFER_SECONDS = 60
BUFFER_SIZE    = int(TARGET_FPS * BUFFER_SECONDS)

EMAIL_LIMIT_REACHED = False
hls_processes = {}


# ============================================================
#  2. DISTANCE ZONE CLASSIFIER
#  Base sa bounding box height ng tao sa frame:
#  NEAR   > 280px  — malapit, malaki sa frame
#  MEDIUM 140-280px — katamtamang distansya
#  FAR    < 140px  — malayo, maliit sa frame
# ============================================================

def get_distance_zone(person_height_px):
    if person_height_px > 280:
        return "NEAR"
    elif person_height_px > 140:
        return "MEDIUM"
    else:
        return "FAR"

def get_distance_thresholds(zone):
    """
    Nagbabalik ng dict ng thresholds na naka-adjust sa distansya.
    Kapag malayo ang tao — mas relaxed ang thresholds para ma-detect pa rin.
    Kapag malapit — mas strict para hindi mag-false positive.
    """
    if zone == "NEAR":
        return {
            # Loitering — mas mabilis mag-flag kapag malapit (30s)
            "loiter_seconds":   30.0,
            # Pacing — kailangan ng mas mahabang path kapag malapit
            "pace_path_mult":   2.0,   # total_path > person_height * 2.0
            "pace_net_ratio":   0.6,   # net_displacement < total_path * 0.6
            "pace_std_x":       8,     # std_x > 8
            # Loitering Area — mas maliit na area ang considered na "loitering"
            "loiter_area_w":    0.25,  # area_width < frame_width * 0.25
            "loiter_area_h":    0.25,
            # Suspicious — kailangan ng mas maraming visible joints
            "min_visible_kpts": 12,
            # Stealing — mas strict ang stashing zone
            "steal_torso_mult": 0.25,
            "steal_min_thresh": 0.03,
        }
    elif zone == "MEDIUM":
        return {
            "loiter_seconds":   35.0,
            "pace_path_mult":   2.5,
            "pace_net_ratio":   0.6,
            "pace_std_x":       6,
            "loiter_area_w":    0.30,
            "loiter_area_h":    0.30,
            "min_visible_kpts": 8,
            "steal_torso_mult": 0.30,
            "steal_min_thresh": 0.03,
        }
    else:  # FAR
        return {
            # Loitering — mas matagal bago mag-flag kapag malayo (45s)
            "loiter_seconds":   45.0,
            # Pacing — mas madaling mag-trigger kapag malayo (mas maliit ang path)
            "pace_path_mult":   1.5,
            "pace_net_ratio":   0.65,
            "pace_std_x":       4,
            # Loitering Area — mas malaki ang area para ma-catch kahit malayo
            "loiter_area_w":    0.40,
            "loiter_area_h":    0.40,
            # Suspicious — mas konting visible joints lang kailangan kapag malayo
            "min_visible_kpts": 6,
            # Stealing — mas relaxed ang stashing zone
            "steal_torso_mult": 0.35,
            "steal_min_thresh": 0.04,
        }


# ============================================================
#  3. ACCURACY CONFIDENCE HELPERS
# ============================================================

def get_confidence_label(accuracy):
    if accuracy >= ACC_HIGH:     return "HIGH CONFIDENCE"
    elif accuracy >= ACC_MEDIUM: return "MODERATE CONFIDENCE"
    else:                        return "LOW CONFIDENCE"

def get_confidence_color(accuracy):
    if accuracy >= ACC_HIGH:     return (0, 255, 0)
    elif accuracy >= ACC_MEDIUM: return (0, 165, 255)
    else:                        return (0, 0, 255)


# ============================================================
#  4. BEHAVIOR VALIDATOR
# ============================================================

class BehaviorValidator:
    def __init__(self):
        self.alert_counters   = {}
        self.MIN_FRAMES_STEAL = 8
        self.MIN_FRAMES_SUSP  = 10

    def get_temporal_validation(self, track_id, current_label):
        if track_id not in self.alert_counters:
            self.alert_counters[track_id] = {"label": "Normal", "count": 0}
        counter = self.alert_counters[track_id]
        if current_label == counter["label"]:
            counter["count"] += 1
        else:
            counter["label"] = current_label
            counter["count"] = 0 if current_label == "Normal" else max(1, counter["count"] // 2)

        if "Stealing" in counter["label"] and counter["count"] >= self.MIN_FRAMES_STEAL:
            return True
        if (any(x in counter["label"] for x in ["Suspicious", "Loitering", "Pacing"])
                and counter["count"] >= self.MIN_FRAMES_SUSP):
            return True
        return False

validator = BehaviorValidator()


# ============================================================
#  5. FLASK + FIREBASE INIT
# ============================================================

app = Flask(__name__)
CORS(app)

if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()


# ============================================================
#  6. HLS STREAM MANAGER
# ============================================================

def start_hls_stream(rtsp_url, camera_name):
    global hls_processes
    if camera_name in hls_processes:
        if hls_processes[camera_name].poll() is None:
            print(f"[HLS] Already running for {camera_name}")
            return
    out_dir  = os.path.join(HLS_DIR, camera_name)
    os.makedirs(out_dir, exist_ok=True)
    playlist = os.path.join(out_dir, "stream.m3u8")
    cmd = [
        "ffmpeg", "-loglevel", "warning",
        "-rtsp_transport", "tcp", "-i", rtsp_url,
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-vf", "scale=1280:-2", "-b:v", "1000k", "-g", "30", "-sc_threshold", "0",
        "-f", "hls", "-hls_time", "2", "-hls_list_size", "5",
        "-hls_flags", "delete_segments+append_list",
        "-hls_segment_filename", os.path.join(out_dir, "seg%03d.ts"),
        playlist
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hls_processes[camera_name] = proc
    print(f"[HLS] ✅ Started for {camera_name}")


def monitor_hls_processes():
    while True:
        time.sleep(10)
        for cam_name, proc in list(hls_processes.items()):
            if proc.poll() is not None:
                print(f"[HLS] ⚠️  Restarting crashed stream for {cam_name}...")
                cam = cameras_dict.get(cam_name)
                if cam:
                    start_hls_stream(cam.src, cam_name)

threading.Thread(target=monitor_hls_processes, daemon=True).start()


# ============================================================
#  7. RTSP VIDEO STREAM CLASS
# ============================================================

class RTSPVideoStream:
    def __init__(self, src, name=None):
        self.src     = src
        self.name    = name
        self.cap     = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.ret     = False
        self.frame   = None
        self.stopped = False
        self.online  = self.cap.isOpened()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.ret = True; self.frame = frame; self.online = True
                else:
                    self.ret = False; self.online = False
                    print(f"⚠️  RTSP lost for {self.name} — reconnecting...")
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            else:
                self.cap.release(); time.sleep(2)
                self.cap    = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                self.online = self.cap.isOpened()
                if self.online:
                    print(f"✅ Camera {self.name} reconnected!")

    def read(self):
        if not self.cap.isOpened():
            self.online = False
            return False, None
        return self.ret, self.frame


# ============================================================
#  8. LOAD CAMERAS FROM FIRESTORE
# ============================================================

def load_cameras_from_firestore():
    cameras = {}
    try:
        for cam_doc in db.collection("cameras").stream():
            data        = cam_doc.to_dict()
            camera_name = data.get("name")
            rtsp_url    = data.get("rtsp_url")
            if not camera_name or not rtsp_url:
                print("⚠️  Skipping invalid camera document:", data)
                continue
            cameras[camera_name] = RTSPVideoStream(rtsp_url, name=camera_name)
            print(f"✅ Loaded camera: {camera_name}")
            start_hls_stream(rtsp_url, camera_name)
    except Exception as e:
        print(f"❌ Failed to load cameras: {e}")
    return cameras

cameras_dict = load_cameras_from_firestore()


# ============================================================
#  9. CLOUDINARY CONFIG
# ============================================================

cloudinary.config(
    cloud_name = "dpkds0mpw",
    api_key    = "573419718341736",
    api_secret = "wFzLdsZ_O9vLiZKP2bvZCN_XGjA",
    secure     = True
)


# ============================================================
#  10. LOAD AI MODELS
# ============================================================

print(">> Loading AI Models...")
try:
    lstm_model     = load_model(ANOMALY_MODEL_PATH)
    yolo_model     = YOLO(YOLO_MODEL)
    stealing_model = load_model(STEALING_MODEL_PATH) if os.path.exists(STEALING_MODEL_PATH) else None
    print(">> Models loaded successfully.")
except Exception as e:
    print(f"!! Model load error: {e}")
    sys.exit(1)


# ============================================================
#  11. HELPER FUNCTIONS
# ============================================================

def get_all_user_emails():
    emails = []
    try:
        for doc in db.collection("users").stream():
            data = doc.to_dict()
            if "email" in data:
                emails.append(data["email"])
    except Exception as e:
        print(f"!! Error fetching emails: {e}")
    return emails


def save_to_firebase(label, cloud_url, accuracy, camera_name,
                     track_id=0, timestamp_str=""):
    try:
        ph_tz      = timezone(timedelta(hours=8))
        confidence = get_confidence_label(accuracy)
        db.collection("detections").add({
            "camera_name":   camera_name,
            "type":          label,
            "video_url":     cloud_url,
            "accuracy":      float(accuracy) / 100.0 if accuracy > 0 else 0.0,
            "timestamp":     datetime.now(ph_tz),
            "created_at":    firestore.SERVER_TIMESTAMP,
            "track_id":      track_id,
            "incident_time": timestamp_str,
            "confidence":    confidence,
        })
        print(f"✅ Saved to Firebase: {label} | {accuracy}% ({confidence})")
    except Exception as e:
        print(f"!! Firebase Error: {e}")


def send_email_alert_immediate(label, track_id, accuracy, camera_name, timestamp):
    global EMAIL_LIMIT_REACHED
    if not ENABLE_EMAILS or EMAIL_LIMIT_REACHED:
        return
    recipient_list = get_all_user_emails()
    if not recipient_list:
        return
    confidence = get_confidence_label(accuracy)
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        for email in recipient_list:
            msg = MIMEMultipart()
            msg['From']    = f"Security System <{EMAIL_SENDER}>"
            msg['To']      = email
            msg['Subject'] = f"🚨 ALERT: {label} Detected! [{confidence}]"
            body = f"""
            <h2>⚠️ Security Alert</h2>
            <table style="border-collapse:collapse;width:100%;font-family:monospace;">
              <tr><td style="padding:6px;color:#888;font-weight:bold;">INCIDENT</td>
                  <td style="padding:6px;font-weight:bold;">{label}</td></tr>
              <tr><td style="padding:6px;color:#888;font-weight:bold;">CAMERA</td>
                  <td style="padding:6px;">{camera_name}</td></tr>
              <tr><td style="padding:6px;color:#888;font-weight:bold;">TIME</td>
                  <td style="padding:6px;">{timestamp}</td></tr>
              <tr><td style="padding:6px;color:#888;font-weight:bold;">TRACK ID</td>
                  <td style="padding:6px;">{track_id}</td></tr>
              <tr><td style="padding:6px;color:#888;font-weight:bold;">ACCURACY</td>
                  <td style="padding:6px;color:{'#7c3aed' if accuracy >= ACC_HIGH else '#d97706'};font-weight:bold;">
                    {accuracy}% — {confidence}</td></tr>
            </table>
            <hr>
            <p style="color:#888;font-size:12px;"><i>
              Video evidence is being uploaded.
              You will receive a follow-up email with the video link shortly.
            </i></p>
            """
            msg.attach(MIMEText(body, 'html'))
            server.send_message(msg)
        server.quit()
        print(f"✅ Instant alert email sent to {len(recipient_list)} user(s).")
    except Exception as e:
        print(f"!! Instant email failed: {e}")
        if "limit exceeded" in str(e) or "5.4.5" in str(e):
            EMAIL_LIMIT_REACHED = True


def send_email_with_video(label, video_url, track_id, accuracy, timestamp):
    global EMAIL_LIMIT_REACHED
    if not ENABLE_EMAILS or EMAIL_LIMIT_REACHED:
        return
    recipient_list = get_all_user_emails()
    if not recipient_list:
        return
    confidence = get_confidence_label(accuracy)
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        for email in recipient_list:
            msg = MIMEMultipart()
            msg['From']    = f"Security System <{EMAIL_SENDER}>"
            msg['To']      = email
            msg['Subject'] = f"📹 Video Evidence Ready: {label} — ID {track_id}"
            body = f"""
            <h2>📹 Video Evidence Available</h2>
            <table style="border-collapse:collapse;width:100%;font-family:monospace;">
              <tr><td style="padding:6px;color:#888;font-weight:bold;">INCIDENT</td>
                  <td style="padding:6px;font-weight:bold;">{label}</td></tr>
              <tr><td style="padding:6px;color:#888;font-weight:bold;">TIME</td>
                  <td style="padding:6px;">{timestamp}</td></tr>
              <tr><td style="padding:6px;color:#888;font-weight:bold;">TRACK ID</td>
                  <td style="padding:6px;">{track_id}</td></tr>
              <tr><td style="padding:6px;color:#888;font-weight:bold;">ACCURACY</td>
                  <td style="padding:6px;color:{'#7c3aed' if accuracy >= ACC_HIGH else '#d97706'};font-weight:bold;">
                    {accuracy}% — {confidence}</td></tr>
            </table>
            <br>
            <a href="{video_url}" style="background:#7c3aed;color:white;padding:12px 24px;
                border-radius:8px;text-decoration:none;font-weight:bold;font-family:sans-serif;">
                ▶ View Video Evidence
            </a>
            """
            msg.attach(MIMEText(body, 'html'))
            server.send_message(msg)
        server.quit()
        print(f"✅ Video link email sent to {len(recipient_list)} user(s).")
    except Exception as e:
        print(f"!! Video email failed: {e}")
        if "limit exceeded" in str(e) or "5.4.5" in str(e):
            EMAIL_LIMIT_REACHED = True


def save_alert_clip(frame_buffer, label, track_id, accuracy=0,
                    fps=30, camera_name=""):
    try:
        frames = list(frame_buffer)
        if not frames:
            return

        ph_tz      = timezone(timedelta(hours=8))
        ts_display = datetime.now(ph_tz).strftime("%Y-%m-%d %H:%M:%S")
        ts_file    = datetime.now(ph_tz).strftime("%Y%m%d_%H%M%S")
        safe_label = label.replace(":", "").replace(" ", "_")
        confidence = get_confidence_label(accuracy)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fn = f"{OUTPUT_DIR}/{camera_name}_{safe_label}_ID{track_id}_{ts_file}.mp4"

        h, w, _ = frames[0].shape
        write_fps = max(fps, 5.0)
        fourcc    = cv2.VideoWriter_fourcc(*'mp4v')
        out       = cv2.VideoWriter(fn, fourcc, write_fps, (w, h))

        if not out.isOpened():
            fn     = fn.replace(".mp4", ".avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out    = cv2.VideoWriter(fn, fourcc, write_fps, (w, h))

        if not out.isOpened():
            print("!! VideoWriter failed entirely.")
            return

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        font_thick = 1
        header_h   = 40

        for f in frames:
            frame_out = f.copy()
            cv2.rectangle(frame_out, (0, h - header_h), (w, h), (0, 0, 0), -1)
            header_text = (f"{ts_display} | ID:{track_id} {label} "
                           f"| ACC:{accuracy}% [{confidence}]")
            (_, text_h), _ = cv2.getTextSize(header_text, font, font_scale, font_thick)
            text_y = h - int((header_h - text_h) / 2) - 5
            cv2.putText(frame_out, header_text, (10, text_y),
                        font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
            out.write(frame_out)

        out.release()

        if ENABLE_EMAILS and not EMAIL_LIMIT_REACHED and accuracy >= EMAIL_MIN_ACCURACY:
            threading.Thread(
                target=send_email_alert_immediate,
                args=(label, track_id, accuracy, camera_name, ts_display)
            ).start()

        new_entry = {
            "camera_name": camera_name,
            "timestamp":   ts_display,
            "type":        label,
            "video_url":   fn,
            "accuracy":    float(accuracy) / 100.0 if accuracy > 0 else 0.0,
            "confidence":  confidence,
        }
        current_logs = []
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, 'r') as f:
                    current_logs = json.load(f)
            except Exception:
                pass
        current_logs.append(new_entry)
        with open(LOG_FILE, 'w') as f:
            json.dump(current_logs[-20:], f, indent=2)

        print(f"✅ Alert saved: ID {track_id} — {label} ({accuracy}% | {confidence})")

        if os.path.exists(fn) and os.path.getsize(fn) > 0:
            print(f"📤 Uploading ID {track_id}...")
            res = None; upload_success = False
            for attempt in range(1, 6):
                try:
                    res = cloudinary.uploader.upload(
                        fn, resource_type="video",
                        folder=f"ai_detections/{camera_name}"
                    )
                    upload_success = True
                    break
                except Exception as e:
                    print(f"!! Upload attempt {attempt} failed: {e}")
                    time.sleep(5)

            if upload_success and res:
                raw_url = res.get("secure_url", "")
                if raw_url:
                    playable_url = raw_url.replace("/upload/", "/upload/vc_h264,f_mp4/")
                    print(f"🔗 URL: {playable_url}")
                    save_to_firebase(label, playable_url, accuracy, camera_name,
                                     track_id=track_id, timestamp_str=ts_display)
                    if ENABLE_EMAILS and not EMAIL_LIMIT_REACHED and accuracy >= EMAIL_MIN_ACCURACY:
                        threading.Thread(
                            target=send_email_with_video,
                            args=(label, playable_url, track_id, accuracy, ts_display)
                        ).start()
            else:
                print("!! Upload failed after all attempts.")

    except Exception as e:
        print(f"!! save_alert_clip error: {e}")


# ── Keypoint Helpers ─────────────────────────────────────────

def check_head_scanning(kpts, scan_history):
    nose_x     = kpts[0]
    left_sh_x  = kpts[10]
    right_sh_x = kpts[12]
    shoulder_w = abs(right_sh_x - left_sh_x)
    if shoulder_w < 0.01:
        return False
    look_ratio = (nose_x - left_sh_x) / shoulder_w

    # ← UPDATED: mas maluwag na ratio para sa eye level camera
    # Dati: 0.15/0.85 — kailangan ng extreme na head turn
    # Bago: 0.30/0.70 — normal na pagtingin sa kaliwa/kanan
    if look_ratio < 0.30:   scan_history.append("LEFT")
    elif look_ratio > 0.70: scan_history.append("RIGHT")
    else:                   scan_history.append("CENTER")

    if len(scan_history) < (SCAN_LEN // 4):
        return False

    # ← UPDATED: 3 beses lang sa bawat side (dati 5)
    return scan_history.count("LEFT") >= 3 and scan_history.count("RIGHT") >= 3


def is_hand_in_stashing_zone(kpts, torso_mult, min_thresh):
    """
    Distance-aware stashing zone — mas relaxed kapag malayo ang tao.
    torso_mult at min_thresh ay galing sa get_distance_thresholds().
    """
    lw_x,    lw_y    = kpts[18], kpts[19]
    rw_x,    rw_y    = kpts[20], kpts[21]
    l_hip_x, l_hip_y = kpts[22], kpts[23]
    r_hip_x, r_hip_y = kpts[24], kpts[25]
    shoulder_y     = (kpts[11] + kpts[13]) / 2
    torso_h        = abs((l_hip_y + r_hip_y) / 2 - shoulder_y)
    dynamic_thresh = max(torso_h * torso_mult, min_thresh)
    if lw_x > r_hip_x and rw_x < l_hip_x:
        return False
    dist_l = np.hypot(lw_x - l_hip_x, lw_y - l_hip_y)
    dist_r = np.hypot(rw_x - r_hip_x, rw_y - r_hip_y)
    return min(dist_l, dist_r) < dynamic_thresh


# ============================================================
#  12. FLASK ROUTES
# ============================================================

@app.route("/addCamera", methods=["POST"])
def add_camera():
    data        = request.get_json()
    user_id     = data.get("userId")
    camera_name = data.get("cameraName")
    rtsp_url    = data.get("rtspUrl")
    if not all([user_id, camera_name, rtsp_url]):
        return {"error": "Missing data"}, 400
    if camera_name in cameras_dict:
        return {"error": "Camera name already exists"}, 400
    if any(c.src == rtsp_url for c in cameras_dict.values()):
        return {"error": "RTSP URL already added"}, 400
    if any(db.collection("cameras").where("name",     "==", camera_name).stream()):
        return {"error": "Camera name already exists in database"}, 400
    if any(db.collection("cameras").where("rtsp_url", "==", rtsp_url).stream()):
        return {"error": "RTSP URL already exists in database"}, 400
    cameras_dict[camera_name] = RTSPVideoStream(rtsp_url, name=camera_name)
    start_hls_stream(rtsp_url, camera_name)
    db.collection("cameras").add({
        "name": camera_name, "rtsp_url": rtsp_url,
        "owner": user_id, "created_at": firestore.SERVER_TIMESTAMP
    })
    print(f"[INFO] Camera added: {camera_name} ({rtsp_url}) by {user_id}")
    return {"message": f"Camera '{camera_name}' added successfully!"}, 200


@app.route('/cameras', methods=['GET'])
def get_cameras():
    return {"cameras": list(cameras_dict.keys())}, 200

@app.route('/video/<camera_name>')
def video_feed(camera_name):
    return Response(gen_frames(camera_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hls/<camera_name>/stream.m3u8')
def hls_playlist(camera_name):
    hls_cam_dir = os.path.join(HLS_DIR, camera_name)
    if not os.path.exists(os.path.join(hls_cam_dir, "stream.m3u8")):
        return {"error": f"HLS stream not ready for {camera_name}"}, 404
    response = send_from_directory(hls_cam_dir, "stream.m3u8")
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/hls/<camera_name>/<path:segment>')
def hls_segment(camera_name, segment):
    hls_cam_dir = os.path.join(HLS_DIR, camera_name)
    response = send_from_directory(hls_cam_dir, segment)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/hls/status')
def hls_status():
    status = {}
    for cam_name, proc in hls_processes.items():
        status[cam_name] = "running" if proc.poll() is None else "stopped"
    return {"hls_streams": status}, 200

@app.route('/logs')
def get_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return Response(f.read(), mimetype='application/json')
    return Response("[]", mimetype='application/json')


# ============================================================
#  13. MAIN DETECTION LOOP
# ============================================================

people_states = {}

def gen_frames(camera_name):
    camera = cameras_dict.get(camera_name)
    if not camera:
        print(f"❌ Camera '{camera_name}' not found.")
        return

    frame_buffer    = deque(maxlen=BUFFER_SIZE)
    frame_count     = 0
    prev_frame_time = 0
    real_time_fps   = 15.0

    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        now = time.time()
        if prev_frame_time > 0:
            real_time_fps = 0.9 * real_time_fps + 0.1 * (1.0 / (now - prev_frame_time))
        prev_frame_time = now

        orig_h, orig_w = frame.shape[:2]
        frame  = cv2.resize(frame, (640, int(orig_h * (640 / orig_w))))
        height, width = frame.shape[:2]

        results = yolo_model.track(frame, persist=True, verbose=False,
                                   classes=[0], conf=0.45)

        if results and results[0].boxes.id is not None:
            boxes       = results[0].boxes.xyxy.cpu().numpy()
            track_ids   = results[0].boxes.id.int().cpu().tolist()
            keypoints   = results[0].keypoints.xyn.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, track_id, kpts, y_conf in zip(boxes, track_ids,
                                                    keypoints, confidences):
                kpts_flat = kpts.flatten()

                if track_id not in people_states:
                    people_states[track_id] = {
                        'pose_seq':              [],
                        'loc_hist':              deque(maxlen=HISTORY_LEN),
                        'scan_hist':             deque(maxlen=SCAN_LEN),
                        'stationary_counter':    0,
                        'stationary_start_time': 0.0,
                        'current_label':         "Normal",
                        'current_color':         (0, 255, 0),
                        'smoothed_box':          box.copy(),
                        'current_acc':           0,
                        'last_lstm_err':         0.0,
                        'last_steal_prob':       0.0,
                        'last_save_time':        0
                    }

                person = people_states[track_id]
                person['smoothed_box'] = 0.5 * box + 0.5 * person['smoothed_box']

                person['pose_seq'].append(kpts_flat)
                if len(person['pose_seq']) > 30:
                    person['pose_seq'].pop(0)

                bx1, by1, bx2, by2 = map(int, person['smoothed_box'])
                cx = (bx1 + bx2) // 2
                cy = (by1 + by2) // 2

                # ── Distance zone — auto-detect base sa bounding box height ──
                person_height_px = max(by2 - by1, 1)
                zone    = get_distance_zone(person_height_px)
                thresh  = get_distance_thresholds(zone)

                dist_speed = 0.0
                if len(person['loc_hist']) > 0:
                    lx, ly     = person['loc_hist'][-1]
                    dist_speed = np.hypot(cx - lx, cy - ly)
                    if dist_speed > 2:
                        person['loc_hist'].append((cx, cy))
                    if dist_speed < 8.0:
                        if person['stationary_start_time'] == 0.0:
                            person['stationary_start_time'] = time.time()
                        person['stationary_counter'] += 1
                        if person['stationary_counter'] > (STILLNESS_LIMIT // 2):
                            person['loc_hist'].clear()
                    else:
                        person['stationary_counter']    = 0
                        person['stationary_start_time'] = 0.0
                else:
                    person['loc_hist'].append((cx, cy))

                if len(person['pose_seq']) == 30 and (frame_count % LOGIC_SKIP == 0):
                    inp        = np.array([person['pose_seq']])
                    lstm_out   = lstm_model.predict(inp, verbose=0)
                    lstm_err   = float(np.mean(np.abs(lstm_out - inp)))
                    steal_prob = float(stealing_model.predict(inp, verbose=0)[0][0]) \
                                 if stealing_model else 0.0
                    person['last_lstm_err']   = lstm_err
                    person['last_steal_prob'] = steal_prob
                else:
                    lstm_err   = person['last_lstm_err']
                    steal_prob = person['last_steal_prob']

                # ── Distance-aware stealing check ────────────────────────────
                is_hand_close = is_hand_in_stashing_zone(
                    kpts_flat,
                    thresh["steal_torso_mult"],
                    thresh["steal_min_thresh"]
                )

                # ── Visible keypoints — relaxed kapag malayo ──────────────────
                visible_kpts = sum(
                    1 for i in range(17)
                    if kpts_flat[i * 2] > 0.01 and kpts_flat[i * 2 + 1] > 0.01
                )

                raw_label = "Normal"
                color     = (0, 255, 0)
                acc       = int(y_conf * 100)

                stationary_seconds = (time.time() - person['stationary_start_time']
                                      if person['stationary_start_time'] > 0 else 0)

                # ── LOITERING (Still) — distance-aware seconds ────────────────
                if stationary_seconds > thresh["loiter_seconds"]:
                    raw_label = "Loitering (Still)"
                    color     = (0, 255, 255)
                    acc       = 90

                elif len(person['loc_hist']) >= (HISTORY_LEN // 2):
                    xs = [p[0] for p in person['loc_hist']]
                    ys = [p[1] for p in person['loc_hist']]
                    total_path       = sum(np.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1])
                                          for i in range(1, len(xs)))
                    net_displacement = np.hypot(xs[-1]-xs[0], ys[-1]-ys[0])
                    area_width       = max(xs) - min(xs)
                    area_height      = max(ys) - min(ys)
                    std_x            = np.std(xs)

                    # ── PACING — distance-aware path/std thresholds ───────────
                    if (total_path > (person_height_px * thresh["pace_path_mult"]) and
                            net_displacement < (total_path * thresh["pace_net_ratio"]) and
                            std_x > thresh["pace_std_x"]):
                        raw_label = "Pacing"
                        color     = (255, 140, 0)
                        acc       = 88

                    # ── LOITERING (Area) — distance-aware area size ───────────
                    elif (area_width  < (width  * thresh["loiter_area_w"]) and
                          area_height < (height * thresh["loiter_area_h"])):
                        raw_label = "Loitering (Area)"
                        color     = (0, 255, 255)
                        acc       = 92

                    # ── STEALING — distance-aware stashing zone ───────────────
                    elif steal_prob > STEAL_THRESH and is_hand_close and dist_speed < 15:
                        raw_label = "PREDICTION: Stealing"
                        color     = (128, 0, 128)
                        acc       = int(steal_prob * 100)

                    elif check_head_scanning(kpts_flat, person['scan_hist']):
                        raw_label = "Suspicious: Scanning"
                        color     = (255, 165, 0)
                        acc       = 85

                    # ── SUSPICIOUS — distance-aware visible keypoints ─────────
                    elif lstm_err > POSE_THRESHOLD and visible_kpts >= thresh["min_visible_kpts"]:
                        raw_label = "Suspicious Behavior"
                        color     = (0, 165, 255)
                        acc       = min(int((lstm_err / 0.5) * 100), 99)

                else:
                    # Fallback — kahit kulang pa ang history
                    if stationary_seconds > thresh["loiter_seconds"]:
                        raw_label = "Loitering (Still)"
                        color     = (0, 255, 255)
                        acc       = 90
                    elif steal_prob > STEAL_THRESH and is_hand_close and dist_speed < 15:
                        raw_label = "PREDICTION: Stealing"
                        color     = (128, 0, 128)
                        acc       = int(steal_prob * 100)
                    elif check_head_scanning(kpts_flat, person['scan_hist']):
                        raw_label = "Suspicious: Scanning"
                        color     = (255, 165, 0)
                        acc       = 85
                    elif lstm_err > POSE_THRESHOLD and visible_kpts >= thresh["min_visible_kpts"]:
                        raw_label = "Suspicious Behavior"
                        color     = (0, 165, 255)
                        acc       = min(int((lstm_err / 0.5) * 100), 99)

                is_validated = validator.get_temporal_validation(track_id, raw_label)
                final_label  = raw_label if is_validated else "Normal"
                final_color  = color     if is_validated else (0, 255, 0)
                final_acc    = acc       if is_validated else int(y_conf * 100)

                person['current_label'] = final_label
                person['current_color'] = final_color
                person['current_acc']   = final_acc

                if final_label != "Normal":
                    cooldown = 5.0 if "Stealing" in final_label else 30.0
                    if (time.time() - person['last_save_time']) > cooldown:
                        person['last_save_time'] = time.time()
                        threading.Thread(
                            target=save_alert_clip,
                            args=(list(frame_buffer), final_label, track_id,
                                  final_acc, real_time_fps, camera_name)
                        ).start()

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), person['current_color'], 2)
                conf_label = get_confidence_label(final_acc)

                # ── Show distance zone sa overlay para ma-verify ──────────────
                cv2.putText(frame, f"ID {track_id} [{zone}]",
                            (bx1, by1 - 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, person['current_color'], 2)
                cv2.putText(frame,
                            f"{person['current_label']} ({person['current_acc']}%)",
                            (bx1, by1 - 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, person['current_color'], 2)
                cv2.putText(frame, conf_label,
                            (bx1, by1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, get_confidence_color(final_acc), 1)

                for i in range(17):
                    xk = int(kpts_flat[i * 2]     * width)
                    yk = int(kpts_flat[i * 2 + 1] * height)
                    if xk > 0 and yk > 0:
                        cv2.circle(frame, (xk, yk), 4, (0, 255, 0), -1)

        frame_count += 1
        frame_buffer.append(frame.copy())
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')


# ============================================================
#  14. ENTRY POINT
# ============================================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)