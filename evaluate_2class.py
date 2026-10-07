
from ultralytics import YOLO

# ==============================
# MODEL
# ==============================

MODEL_PATH = "models/best_person_bag.pt"

# ==============================
# LOAD MODEL
# ==============================

print("Loading trained model...")

model = YOLO(MODEL_PATH)

# ==============================
# EVALUATE
# ==============================

print()
print("================================")
print("EVALUATING PERSON + BAG MODEL")
print("================================")

results = model.val(
    data="data/dataset_2class.yaml",
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    plots=True
)

# ==============================
# DISPLAY RESULTS
# ==============================

print()
print("================================")
print("EVALUATION COMPLETE")
print("================================")

print(f"Precision : {results.box.mp:.3f}")
print(f"Recall    : {results.box.mr:.3f}")
print(f"mAP50     : {results.box.map50:.3f}")
print(f"mAP50-95  : {results.box.map:.3f}")

print()
print("Class-wise results:")

for i, name in enumerate(model.names):

    print(f"{name}:")
    print(f"  Precision : {results.box.p[i]:.3f}")
    print(f"  Recall    : {results.box.r[i]:.3f}")
    print(f"  mAP50     : {results.box.ap50[i]:.3f}")
    print(f"  mAP50-95  : {results.box.ap[i]:.3f}")

print()
print("Evaluation plots are saved inside runs/detect/")
