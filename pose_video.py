from ultralytics import YOLO

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Deteksi pose pada video
results = model("video_input.mp4", save=True, show=True)