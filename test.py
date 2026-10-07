
from ultralytics import YOLO
import cv2
from pathlib import Path

# ==============================
# SETTINGS
# ==============================

MODEL_PATH = "models/best_person_bag.pt"
VIDEO_PATH = "videos/vid3.mp4"
OUTPUT_PATH = "outputs/test_result.mp4"

CONFIDENCE = 0.20

# ==============================
# LOAD MODEL
# ==============================

print("Loading trained model...")
model = YOLO(MODEL_PATH)

# ==============================
# OPEN VIDEO
# ==============================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video FPS:", fps)
print("Resolution:", width, "x", height)

# ==============================
# OUTPUT FOLDER
# ==============================

Path("outputs").mkdir(exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

# ==============================
# CREATE WINDOW
# ==============================

cv2.namedWindow(
    "AI Person + Bag Detection",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "AI Person + Bag Detection",
    1200,
    700
)

print()
print("================================")
print("LIVE DETECTION STARTED")
print("Press Q to stop")
print("================================")

# ==============================
# PROCESS VIDEO
# ==============================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # YOLO detection
    results = model.predict(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    person_count = 0
    bag_count = 0

    # ==============================
    # DRAW DETECTIONS
    # ==============================

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            # PERSON
            if class_id == 0:

                person_count += 1

                label = f"PERSON {confidence:.2f}"

                # Green
                color = (0, 255, 0)

            # BAG
            elif class_id == 1:

                bag_count += 1

                label = f"BAG {confidence:.2f}"

                # Blue
                color = (255, 0, 0)

            else:
                continue

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # Label background
            label_width = 220

            cv2.rectangle(
                frame,
                (x1, max(0, y1 - 40)),
                (x1 + label_width, y1),
                color,
                -1
            )

            # Label
            cv2.putText(
                frame,
                label,
                (x1 + 5, max(25, y1 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    # ==============================
    # INFORMATION PANEL
    # ==============================

    cv2.rectangle(
        frame,
        (10, 10),
        (350, 125),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "AI PERSON + BAG DETECTION",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Persons: {person_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Bags: {bag_count}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 0),
        2
    )

    # ==============================
    # SHOW LIVE WINDOW
    # ==============================

    cv2.imshow(
        "AI Person + Bag Detection",
        frame
    )

    # ==============================
    # SAVE VIDEO
    # ==============================

    out.write(frame)

    # ==============================
    # KEYBOARD CONTROL
    # ==============================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print()
        print("Q pressed. Stopping...")
        break

# ==============================
# CLEANUP
# ==============================

cap.release()
out.release()

cv2.destroyAllWindows()

print()
print("================================")
print("TEST COMPLETE")
print("================================")
print("Frames processed:", frame_number)
print("Output:", OUTPUT_PATH)

