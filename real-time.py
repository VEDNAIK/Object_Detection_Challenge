from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Load YOLOv8 model
model_v8 = YOLO("yolov8n.pt")

@app.route('/')
def index():
    return render_template('real-time.html')

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('video_frame')
def handle_video_frame(data):
    print("Frame received from client")
    img_data = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model_v8(frame)
    annotated_frame = results[0].plot() if len(results[0].boxes) > 0 else frame
    print("Frame processed and annotated")

    _, buffer = cv2.imencode('.jpg', annotated_frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    emit('response_frame', {'frame': 'data:image/jpeg;base64,' + frame_b64})

    time.sleep(0.1)

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
