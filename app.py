from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import re

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model = YOLO("yolov8n.pt")  # Example: pretrained YOLO model

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(data):
    # Extract base64 image data
    image_data = re.sub('^data:image/.+;base64,', '', data)
    frame_bytes = base64.b64decode(image_data)
    
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    results = model(frame)
    detections = []

    for r in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = r
        label = model.names[int(cls)]
        if label in ["person", "cell phone"]:  # Example target classes
            detections.append({
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "label": label,
                "confidence": float(conf)
            })

    emit('detection', detections)

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 10000))
    socketio.run(app, host='0.0.0.0', port=port)
