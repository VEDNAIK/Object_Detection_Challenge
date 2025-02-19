from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import base64
from helper import *
from constants import *

app = Flask(__name__)
socketio = SocketIO(app)

# Load YOLOv8 model
model_v8 = YOLO("./models/yolov8n.pt")
# Load YOLOv7 model
model_v7 = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model='./models/yolov7-d6.pt', source='github')
# Load OWL-ViT model
owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files or 'model' not in request.form:
        return jsonify({'error': 'No file or model selected'}), 400

    file = request.files['image']
    selected_model = request.form['model']
    image = Image.open(file).convert("RGB")
    np_image = np.array(image)
    confidence_scores = []

    def detect_yolo(img_array, model):
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        detections = []
        for result in results.xyxy[0]:
            x_min, y_min, x_max, y_max, confidence, class_id = result
            detections.append({
                "box": [x_min.item(), y_min.item(), x_max.item(), y_max.item()],
                "confidence": confidence.item(),
                "label": model.names[int(class_id)]
            })
        return detections

    def detect_owlvit(image_pil):
        inputs = owlvit_processor(text=combined_custom_labels, images=image_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = owlvit_model(**inputs)
        target_sizes = torch.Tensor([image_pil.size[::-1]])
        results = owlvit_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)
        detections = []
        for box, label, score in zip(results[0]["boxes"], results[0]["labels"], results[0]["scores"]):
            if score > 0.1:
                detections.append({
                    "box": [box[0].item(), box[1].item(), box[2].item(), box[3].item()],
                    "confidence": score.item(),
                    "label": combined_custom_labels[label]
                })
        return detections

    if selected_model == 'combined':
        yolo_detections = detect_yolo(np_image, model_v7)
        owlvit_detections = detect_owlvit(image)
        merged_detections = merge_detections(yolo_detections, owlvit_detections)

        for det in merged_detections:
            x_min, y_min, x_max, y_max = map(int, det["box"])
            label = f"{det['label']} {det['confidence']:.2f}"
            confidence_scores.append({'label': det['label'], 'confidence': f"{det['confidence']*100:.1f}"})
            cv2.rectangle(np_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(np_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        annotated_image = np_image

    elif selected_model == 'yolov8':
        results = model_v8(np_image)
        annotated_image = results[0].plot() if len(results[0].boxes) > 0 else np_image
        for det in results[0].boxes:
            confidence_scores.append({'label': model_v8.names[int(det[5].item())], 'confidence': f"{det[4].item()*100:.1f}"})

    elif selected_model == 'yolov7':
        yolo_detections = detect_yolo(np_image, model_v7)
        for det in yolo_detections:
            x_min, y_min, x_max, y_max = map(int, det["box"])
            label = f"{det['label']} {det['confidence']:.2f}"
            confidence_scores.append({'label': det['label'], 'confidence': f"{det['confidence']*100:.1f}"})
            cv2.rectangle(np_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(np_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        annotated_image = np_image

    elif selected_model == 'owlvit':
        owlvit_detections = detect_owlvit(image)
        for det in owlvit_detections:
            x_min, y_min, x_max, y_max = map(int, det["box"])
            label = f"{det['label']} {det['confidence']:.2f}"
            confidence_scores.append({'label': det['label'], 'confidence': f"{det['confidence']*100:.1f}"})
            cv2.rectangle(np_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(np_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        annotated_image = np_image

    else:
        return jsonify({'error': 'Model not supported'}), 400

    _, buffer = cv2.imencode('.jpg', annotated_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'processed_image': f'data:image/jpeg;base64,{img_base64}', 'confidence_scores': confidence_scores})

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)
