from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n-pose.pt')  # Use a pretrained pose model

# Train the model on your dataset (with pretrained weights)
model.train(
    data='dataset/dataset.yaml', 
    epochs=100,
    imgsz=640,
    batch=4,
    augment=True
)
