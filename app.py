from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import threading
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")  # Lightweight, pretrained on COCO dataset
TARGET_CLASSES = ["person", "cell phone"]  # Face detection uses 'person' for head/shoulders, phones are 'cell phone'

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame)
        detections = []

        for r in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = r
            label = model.names[int(cls)]
            if label in TARGET_CLASSES and conf > 0.4:
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                
                # Draw box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detections.append({
                    "x": x, "y": y, "w": w, "h": h,
                    "label": label,
                    "confidence": float(conf)
                })

        # Send detections to frontend
        socketio.emit('detection', detections)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    socketio.run(app, debug=True)
