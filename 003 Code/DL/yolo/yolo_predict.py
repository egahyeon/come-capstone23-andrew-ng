from ultralytics import YOLO

model = YOLO("/home/com_2/workspace/yjshin/capstone/results/runs/detect/train2/weights/best.pt")

results = model.predict(source="/home/com_2/workspace/pig_video/pig_lsw.mp4", save=True)

for r in results:
    print(r.boxes)