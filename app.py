
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # <--- CRITICAL FIX: This stops the network blocks you see in the screenshot

cameras_dict = {} # This stores your cameraName and rtspUrl

@app.route("/addCamera", methods=["POST"])
def add_camera():
    data = request.get_json()
    name = data.get("cameraName")
    url = data.get("rtspUrl")
    
    if name and url:
        # Initialize the camera in your dictionary
        cameras_dict[name] = url 
        print(f"Camera {name} added for RTSP: {url}")
        return jsonify({"success": True}), 200
    return jsonify({"error": "Invalid data"}), 400