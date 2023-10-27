from ultralytics import YOLO

# Load a model
model = YOLO('yolov8l.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='pig.yaml', epochs=100, imgsz=640, device=[2], optimizer='AdamW', lr0=0.01)
#model.train(data='pig.yaml', epochs=150, imgsz=640, device=[2], optimizer='AdamW', lr0=0.02)
