
import cv2
import math
import os
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/best_person_bag.pt"
POSE_MODEL_PATH = "yolov8n-pose.pt"
HAND_MODEL_PATH = "models/hand_landmarker.task"

VIDEO_PATH = "videos/bag4.mp4"
OUTPUT_PATH = "outputs/person_bag_movement.mp4"

# Detection
CONFIDENCE = 0.25

# Hand / bag relationship
HAND_BAG_DISTANCE = 300

# Movement thresholds
BAG_MOVEMENT_THRESHOLD = 4
PERSON_MOVEMENT_THRESHOLD = 4

# Number of frames required for events
PICKING_FRAMES = 4
PICKED_FRAMES = 4
CARRYING_FRAMES = 6
KEPT_FRAMES = 10
PLACED_FRAMES = 10

# Memory
HAND_MEMORY_FRAMES = 60
BAG_MEMORY_FRAMES = 30
PERSON_MEMORY_FRAMES = 15


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def center_of_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def box_distance(box1, box2):
    c1 = center_of_box(box1)
    c2 = center_of_box(box2)
    return distance(c1, c2)


def movement_amount(previous, current):
    if previous is None or current is None:
        return 0

    return distance(previous, current)


def nearest_bag(bags, previous_center):
    """
    If multiple bags are detected, choose the one closest
    to the previous bag position.
    """

    if not bags:
        return None

    if previous_center is None:
        return bags[0]

    best_bag = None
    best_distance = float("inf")

    for bag in bags:
        current_center = center_of_box(bag)

        d = distance(previous_center, current_center)

        if d < best_distance:
            best_distance = d
            best_bag = bag

    return best_bag


def get_hand_points(result, width, height):
    """
    Convert MediaPipe hand landmarks into pixel coordinates.
    """

    points = []

    if result is None:
        return points

    for hand in result.hand_landmarks:

        hand_points = []

        for landmark in hand:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            hand_points.append((x, y))

        points.append(hand_points)

    return points


def nearest_hand_distance(hand_points, bag_center):
    """
    Find the nearest hand landmark to the bag.
    """

    if not hand_points or bag_center is None:
        return None

    minimum = float("inf")

    for hand in hand_points:

        for point in hand:

            d = distance(point, bag_center)

            if d < minimum:
                minimum = d

    return minimum


def get_gesture(hand_points):
    """
    Very simple gesture estimation.

    This is intentionally basic.
    It is used only as supporting information,
    not as the main pickup decision.
    """

    if not hand_points:
        return "UNKNOWN"

    # Use first detected hand
    hand = hand_points[0]

    if len(hand) < 21:
        return "UNKNOWN"

    wrist = hand[0]

    fingers = [
        (8, 5),    # index
        (12, 9),   # middle
        (16, 13),  # ring
        (20, 17)   # pinky
    ]

    extended = 0

    for tip, base in fingers:

        tip_distance = distance(hand[tip], wrist)
        base_distance = distance(hand[base], wrist)

        if tip_distance > base_distance * 1.15:
            extended += 1

    if extended <= 1:
        return "GRAB"

    if extended >= 3:
        return "OPEN"

    return "PARTIAL"


def draw_hand(frame, hand_points):
    """
    Draw MediaPipe hand skeleton.
    """

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    for hand in hand_points:

        for point in hand:
            cv2.circle(
                frame,
                point,
                4,
                (0, 255, 255),
                -1
            )

        for a, b in connections:

            cv2.line(
                frame,
                hand[a],
                hand[b],
                (255, 255, 0),
                2
            )


def draw_pose(frame, pose_result):
    """
    Draw YOLO pose skeleton.
    """

    if pose_result is None:
        return

    for result in pose_result:

        if result.keypoints is None:
            continue

        if result.keypoints.xy is None:
            continue

        keypoints = result.keypoints.xy.cpu().numpy()

        for person_points in keypoints:

            # Draw joints
            for x, y in person_points:

                if x <= 0 or y <= 0:
                    continue

                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    4,
                    (255, 0, 255),
                    -1
                )

            # COCO skeleton connections
            skeleton = [
                (5, 6),
                (5, 7),
                (7, 9),
                (6, 8),
                (8, 10),
                (5, 11),
                (6, 12),
                (11, 12),
                (11, 13),
                (13, 15),
                (12, 14),
                (14, 16)
            ]

            for a, b in skeleton:

                if a >= len(person_points):
                    continue

                if b >= len(person_points):
                    continue

                x1, y1 = person_points[a]
                x2, y2 = person_points[b]

                if x1 <= 0 or y1 <= 0:
                    continue

                if x2 <= 0 or y2 <= 0:
                    continue

                cv2.line(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 255),
                    2
                )


def draw_box(frame, box, label, color):

    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    text_y = max(30, y1 - 10)

    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("======================================")
print("LOADING AI MODELS")
print("======================================")

print("Loading person + bag model...")
model = YOLO(MODEL_PATH)

print("Loading pose model...")
pose_model = YOLO(POSE_MODEL_PATH)

print("Loading MediaPipe hand model...")

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

hand_detector = HandLandmarker.create_from_options(
    hand_options
)

print("ALL MODELS READY")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print()
    print("ERROR: Could not open video:")
    print(VIDEO_PATH)

    hand_detector.close()
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# ============================================================
# OUTPUT VIDEO
# ============================================================

Path("outputs").mkdir(exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# OPENCV DISPLAY WINDOW
# ============================================================

WINDOW_NAME = "PERSON + BAG AI"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1200,
    700
)


# ============================================================
# STATE VARIABLES
# ============================================================

state = "WAITING"

frame_number = 0

previous_bag_center = None
previous_person_center = None

bag_center = None
person_center = None

last_bag_box = None

hand_points = []

last_hand_points = []
hand_missing_frames = 0

bag_missing_frames = 0
person_missing_frames = 0

previous_bag_visible = False
previous_person_visible = False


# Movement
bag_movement = 0
person_movement = 0

total_bag_movement = 0
total_person_movement = 0


# Event counters
picking_counter = 0
picked_counter = 0
carrying_counter = 0
kept_counter = 0
placed_counter = 0


# State history
was_picked = False
was_carried = False
was_kept = False


# ============================================================
# START
# ============================================================

print()
print("======================================")
print("PERSON + BAG MOVEMENT STARTED")
print("Press Q to stop")
print("======================================")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO PERSON + BAG DETECTION
    # --------------------------------------------------------

    results = model.predict(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    persons = []
    bags = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            detected_box = (
                x1,
                y1,
                x2,
                y2
            )

            if class_id == 0:

                persons.append(
                    (
                        detected_box,
                        confidence
                    )
                )

            elif class_id == 1:

                bags.append(
                    (
                        detected_box,
                        confidence
                    )
                )


    # --------------------------------------------------------
    # PERSON SELECTION
    # --------------------------------------------------------

    current_person_box = None

    if persons:

        # Choose largest person
        persons.sort(
            key=lambda item:
            (item[0][2] - item[0][0]) *
            (item[0][3] - item[0][1]),
            reverse=True
        )

        current_person_box = persons[0][0]

        person_center = center_of_box(
            current_person_box
        )

        person_missing_frames = 0

    else:

        person_missing_frames += 1

        if (
            previous_person_center is not None
            and person_missing_frames <= PERSON_MEMORY_FRAMES
        ):
            person_center = previous_person_center

        else:
            person_center = None


    # --------------------------------------------------------
    # BAG SELECTION
    # --------------------------------------------------------

    bag_boxes = [item[0] for item in bags]

    selected_bag = nearest_bag(
        bag_boxes,
        previous_bag_center
    )

    if selected_bag is not None:

        last_bag_box = selected_bag

        bag_center = center_of_box(
            selected_bag
        )

        bag_missing_frames = 0

    else:

        bag_missing_frames += 1

        if (
            last_bag_box is not None
            and bag_missing_frames <= BAG_MEMORY_FRAMES
        ):

            bag_center = center_of_box(
                last_bag_box
            )

        else:

            bag_center = None


    # --------------------------------------------------------
    # MOVEMENT CALCULATION
    # --------------------------------------------------------

    bag_movement = 0
    person_movement = 0

    if bag_center is not None:

        if previous_bag_center is not None:

            bag_movement = movement_amount(
                previous_bag_center,
                bag_center
            )

            if bag_movement < 2:
                bag_movement = 0

        total_bag_movement += bag_movement


    if person_center is not None:

        if previous_person_center is not None:

            person_movement = movement_amount(
                previous_person_center,
                person_center
            )

            if person_movement < 2:
                person_movement = 0

        total_person_movement += person_movement


    # --------------------------------------------------------
    # MEDIAPIPE HAND DETECTION
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    hand_result = hand_detector.detect(
        mp_image
    )

    detected_hands = get_hand_points(
        hand_result,
        width,
        height
    )

    if detected_hands:

        hand_points = detected_hands

        last_hand_points = detected_hands

        hand_missing_frames = 0

    else:

        hand_missing_frames += 1

        if hand_missing_frames <= HAND_MEMORY_FRAMES:

            hand_points = last_hand_points

        else:

            hand_points = []


    # --------------------------------------------------------
    # HAND / BAG DISTANCE
    # --------------------------------------------------------

    hand_bag_distance = None

    if (
        bag_center is not None
        and hand_points
    ):

        hand_bag_distance = nearest_hand_distance(
            hand_points,
            bag_center
        )


    # --------------------------------------------------------
    # GESTURE
    # --------------------------------------------------------

    gesture = get_gesture(
        hand_points
    )


    # --------------------------------------------------------
    # RELATIONSHIP CONDITIONS
    # --------------------------------------------------------

    bag_near_hand = False

    if hand_bag_distance is not None:

        if hand_bag_distance <= HAND_BAG_DISTANCE:

            bag_near_hand = True


    bag_moving = (
        bag_movement >= BAG_MOVEMENT_THRESHOLD
    )

    person_moving = (
        person_movement >= PERSON_MOVEMENT_THRESHOLD
    )

    bag_stationary = (
        bag_movement < BAG_MOVEMENT_THRESHOLD
    )


    # ========================================================
    # EVENT STATE MACHINE
    # ========================================================

    # --------------------------------------------------------
    # 1. WAITING
    # --------------------------------------------------------

    if state == "WAITING":

        if bag_center is not None:

            state = "BAG DETECTED"


    # --------------------------------------------------------
    # 2. BAG DETECTED
    # --------------------------------------------------------

    elif state == "BAG DETECTED":

        if (
            bag_center is not None
            and person_center is not None
            and bag_near_hand
        ):

            picking_counter += 1

        else:

            picking_counter = max(
                0,
                picking_counter - 1
            )


        if picking_counter >= PICKING_FRAMES:

            state = "PERSON PICKING BAG"

            picking_counter = 0


    # --------------------------------------------------------
    # 3. PERSON PICKING BAG
    # --------------------------------------------------------

    elif state == "PERSON PICKING BAG":

        pickup_signal = (
            bag_near_hand
            and (
                bag_moving
                or person_moving
                or gesture == "GRAB"
            )
        )

        if pickup_signal:

            picked_counter += 1

        else:

            picked_counter = max(
                0,
                picked_counter - 1
            )


        if picked_counter >= PICKED_FRAMES:

            state = "PERSON PICKED BAG"

            was_picked = True

            picked_counter = 0


    # --------------------------------------------------------
    # 4. PERSON PICKED BAG
    # --------------------------------------------------------

    elif state == "PERSON PICKED BAG":

        if (
            bag_center is not None
            and person_center is not None
        ):

            carrying_counter += 1

        else:

            carrying_counter = max(
                0,
                carrying_counter - 1
            )


        if carrying_counter >= CARRYING_FRAMES:

            state = "PERSON CARRYING BAG"

            was_carried = True

            carrying_counter = 0


    # --------------------------------------------------------
    # 5. PERSON CARRYING BAG
    # --------------------------------------------------------

    elif state == "PERSON CARRYING BAG":

        if bag_center is None:

            # Do NOT call it placed.
            # Just wait for temporary detection recovery.

            pass

        elif bag_stationary:

            kept_counter += 1

        else:

            kept_counter = max(
                0,
                kept_counter - 1
            )


        if kept_counter >= KEPT_FRAMES:

            state = "PERSON KEPT BAG"

            was_kept = True

            kept_counter = 0


    # --------------------------------------------------------
    # 6. PERSON KEPT BAG
    # --------------------------------------------------------

    elif state == "PERSON KEPT BAG":

        # Bag must remain stationary.
        # Person should move away from it.

        if (
            bag_center is not None
            and person_center is not None
            and bag_stationary
            and person_moving
            and not bag_near_hand
        ):

            placed_counter += 1

        else:

            placed_counter = max(
                0,
                placed_counter - 1
            )


        if placed_counter >= PLACED_FRAMES:

            state = "BAG PLACED"

            placed_counter = 0


    # --------------------------------------------------------
    # 7. BAG PLACED
    # --------------------------------------------------------

    elif state == "BAG PLACED":

        # Keep this state.
        pass


    # ========================================================
    # DRAW PERSON BOXES
    # ========================================================

    for box, confidence in persons:

        label = f"PERSON {confidence:.2f}"

        draw_box(
            frame,
            box,
            label,
            (0, 255, 0)
        )


    # ========================================================
    # DRAW BAG BOXES
    # ========================================================

    for box, confidence in bags:

        label = f"BAG {confidence:.2f}"

        draw_box(
            frame,
            box,
            label,
            (255, 0, 0)
        )


    # ========================================================
    # DRAW POSE
    # ========================================================

    try:

        pose_results = pose_model.predict(
            frame,
            conf=0.35,
            verbose=False
        )

        draw_pose(
            frame,
            pose_results
        )

    except Exception:

        pass


    # ========================================================
    # DRAW HANDS
    # ========================================================

    draw_hand(
        frame,
        hand_points
    )


    # ========================================================
    # DRAW HAND-BAG CONNECTION
    # ========================================================

    if (
        bag_center is not None
        and hand_points
    ):

        nearest_point = None
        nearest_distance_value = float("inf")

        for hand in hand_points:

            for point in hand:

                d = distance(
                    point,
                    bag_center
                )

                if d < nearest_distance_value:

                    nearest_distance_value = d
                    nearest_point = point


        if nearest_point is not None:

            cv2.line(
                frame,
                nearest_point,
                bag_center,
                (0, 255, 255),
                2
            )


    # ========================================================
    # STATE DISPLAY
    # ========================================================

    if state == "BAG DETECTED":
        state_color = (255, 255, 0)

    elif state == "PERSON PICKING BAG":
        state_color = (0, 165, 255)

    elif state == "PERSON PICKED BAG":
        state_color = (0, 255, 255)

    elif state == "PERSON CARRYING BAG":
        state_color = (0, 255, 0)

    elif state == "PERSON KEPT BAG":
        state_color = (255, 165, 0)

    elif state == "BAG PLACED":
        state_color = (255, 0, 255)

    else:
        state_color = (255, 255, 255)


    # ========================================================
    # TOP INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (470, 245),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "AI PERSON + BAG MOVEMENT",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"STATE: {state}",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        state_color,
        2
    )

    if hand_bag_distance is None:
        distance_text = "N/A"
    else:
        distance_text = f"{hand_bag_distance:.0f} px"

    cv2.putText(
        frame,
        f"Hand-Bag Distance: {distance_text}",
        (20, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Bag Movement: {bag_movement:.1f}",
        (20, 158),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Person Movement: {person_movement:.1f}",
        (20, 186),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    hand_status = (
        "LIVE"
        if hand_missing_frames == 0
        else "MEMORY"
    )

    cv2.putText(
        frame,
        f"HAND: {hand_status}",
        (20, 214),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )


    # ========================================================
    # EVENT MESSAGE
    # ========================================================

    event_message = state

    cv2.rectangle(
        frame,
        (10, height - 75),
        (width - 10, height - 10),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        event_message,
        (30, height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        state_color,
        3
    )


    # ========================================================
    # FRAME NUMBER
    # ========================================================

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (width - 180, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    out.write(frame)


    # ========================================================
    # LIVE DISPLAY
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print()
        print("Q PRESSED - STOPPING...")
        break


    # ========================================================
    # UPDATE PREVIOUS POSITIONS
    # ========================================================

    if bag_center is not None:
        previous_bag_center = bag_center

    if person_center is not None:
        previous_person_center = person_center


# ============================================================
# CLEANUP
# ============================================================

cap.release()

out.release()

hand_detector.close()

cv2.destroyAllWindows()

cv2.waitKey(1)


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("======================================")
print("PROCESSING COMPLETE")
print("======================================")

print(f"Frames processed: {frame_number}")
print(f"Final state: {state}")
print(f"Output: {OUTPUT_PATH}")

print()
print("EVENT SUMMARY")
print("--------------------------------------")
print(f"Bag detected: {'YES' if last_bag_box is not None else 'NO'}")
print(f"Person picked bag: {'YES' if was_picked else 'NO'}")
print(f"Person carried bag: {'YES' if was_carried else 'NO'}")
print(f"Person kept bag: {'YES' if was_kept else 'NO'}")
print(f"Bag placed: {'YES' if state == 'BAG PLACED' else 'NO'}")
print("--------------------------------------")
