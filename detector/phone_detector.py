from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")  # Use a fine-tuned version if possible

def detect_phone(frame):
    results = model.predict(source=frame, conf=0.5, verbose=False)
    for r in results:
        for i in range(len(r.boxes.cls)):
            if r.names[int(r.boxes.cls[i])] == 'cell phone':
                return True
    return False
