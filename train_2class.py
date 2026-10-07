from ultralytics import YOLO

print("Loading YOLO11n...")

model = YOLO("yolo11n.pt")

print("Starting training...")

model.train(
    data="data/dataset_2class.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    patience=15,
    device="cpu",
    workers=0,
    project="runs",
    name="person_bag_2class"
)

print()
print("================================")
print("TRAINING COMPLETE")
print("================================")