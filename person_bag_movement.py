import cv2
import math
from pathlib import Path

import mediapipe as mp
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_person_bag.pt"
POSE_MODEL_PATH = BASE_DIR / "models" / "yolov8n-pose.pt"
HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE = 0.25

HAND_BAG_DISTANCE = 300

BAG_MOVEMENT_THRESHOLD = 4
PERSON_MOVEMENT_THRESHOLD = 4

PICKING_FRAMES = 4
PICKED_FRAMES = 4
CARRYING_FRAMES = 6
KEPT_FRAMES = 10
PLACED_FRAMES = 10

HAND_MEMORY_FRAMES = 30
BAG_MEMORY_FRAMES = 20


# ============================================================
# HELPERS
# ============================================================

def center_of_box(box):
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) // 2,
        (y1 + y2) // 2
    )


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def movement(previous, current):

    if previous is None or current is None:
        return 0

    return distance(previous, current)


def nearest_bag(bags, previous_center):

    if not bags:
        return None

    if previous_center is None:
        return bags[0]

    best = None
    best_distance = float("inf")

    for bag in bags:

        c = center_of_box(bag)

        d = distance(
            previous_center,
            c
        )

        if d < best_distance:
            best_distance = d
            best = bag

    return best


# ============================================================
# HAND FUNCTIONS
# ============================================================

def get_hand_points(result, width, height):

    hands = []

    if result is None:
        return hands

    for hand in result.hand_landmarks:

        points = []

        for landmark in hand:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            points.append(
                (x, y)
            )

        hands.append(points)

    return hands


def nearest_hand_distance(
    hands,
    bag_center
):

    if not hands or bag_center is None:
        return None

    minimum = float("inf")

    for hand in hands:

        for point in hand:

            d = distance(
                point,
                bag_center
            )

            if d < minimum:
                minimum = d

    return minimum


def get_gesture(hands):

    if not hands:
        return "UNKNOWN"

    hand = hands[0]

    if len(hand) < 21:
        return "UNKNOWN"

    wrist = hand[0]

    fingers = [
        (8, 5),
        (12, 9),
        (16, 13),
        (20, 17)
    ]

    extended = 0

    for tip, base in fingers:

        tip_distance = distance(
            hand[tip],
            wrist
        )

        base_distance = distance(
            hand[base],
            wrist
        )

        if tip_distance > base_distance * 1.15:
            extended += 1

    if extended <= 1:
        return "GRAB"

    if extended >= 3:
        return "OPEN"

    return "PARTIAL"


# ============================================================
# DRAW HAND
# ============================================================

def draw_hands(frame, hands):

    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20)
    ]

    for hand in hands:

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


# ============================================================
# DRAW POSE
# ============================================================

def draw_pose(frame, pose_results):

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

    for result in pose_results:

        if result.keypoints is None:
            continue

        points = result.keypoints.xy.cpu().numpy()

        for person in points:

            for x, y in person:

                if x > 0 and y > 0:

                    cv2.circle(
                        frame,
                        (int(x), int(y)),
                        4,
                        (255, 0, 255),
                        -1
                    )

            for a, b in skeleton:

                if a >= len(person):
                    continue

                if b >= len(person):
                    continue

                x1, y1 = person[a]
                x2, y2 = person[b]

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


# ============================================================
# DRAW BOX
# ============================================================

def draw_box(
    frame,
    box,
    label,
    color
):

    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    cv2.putText(
        frame,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

def process_video(
    input_path,
    output_path
):

    print("Loading trained YOLO model...")

    model = YOLO(
        str(MODEL_PATH)
    )

    print("Loading pose model...")

    pose_model = YOLO(
        str(POSE_MODEL_PATH)
    )

    print("Loading MediaPipe...")

    BaseOptions = mp.tasks.BaseOptions

    VisionRunningMode = (
        mp.tasks.vision.RunningMode
    )

    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )

    HandLandmarker = (
        mp.tasks.vision.HandLandmarker
    )

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(
                HAND_MODEL_PATH
            )
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2
    )

    hand_detector = (
        HandLandmarker.create_from_options(
            options
        )
    )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():

        hand_detector.close()

        raise RuntimeError(
            "Could not open input video"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    state = "WAITING"

    previous_bag_center = None
    previous_person_center = None

    last_bag_box = None

    hand_points = []
    last_hand_points = []

    hand_missing = 0
    bag_missing = 0

    picking_counter = 0
    picked_counter = 0
    carrying_counter = 0
    kept_counter = 0
    placed_counter = 0

    was_picked = False
    was_carried = False
    was_kept = False

    events = []

    frame_number = 0

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ====================================================
        # YOLO PERSON + BAG
        # ====================================================

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

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                box_data = (
                    x1,
                    y1,
                    x2,
                    y2
                )

                if class_id == 0:

                    persons.append(
                        (
                            box_data,
                            confidence
                        )
                    )

                elif class_id == 1:

                    bags.append(
                        (
                            box_data,
                            confidence
                        )
                    )

        # ====================================================
        # PERSON
        # ====================================================

        person_box = None
        person_center = None

        if persons:

            persons.sort(
                key=lambda item:
                (item[0][2] - item[0][0]) *
                (item[0][3] - item[0][1]),
                reverse=True
            )

            person_box = persons[0][0]

            person_center = center_of_box(
                person_box
            )

        # ====================================================
        # BAG
        # ====================================================

        bag_boxes = [
            item[0]
            for item in bags
        ]

        selected_bag = nearest_bag(
            bag_boxes,
            previous_bag_center
        )

        bag_center = None

        if selected_bag is not None:

            last_bag_box = selected_bag

            bag_center = center_of_box(
                selected_bag
            )

            bag_missing = 0

        else:

            bag_missing += 1

            if (
                last_bag_box is not None
                and bag_missing <= BAG_MEMORY_FRAMES
            ):

                bag_center = center_of_box(
                    last_bag_box
                )

        # ====================================================
        # MOVEMENT
        # ====================================================

        bag_movement = movement(
            previous_bag_center,
            bag_center
        )

        person_movement = movement(
            previous_person_center,
            person_center
        )

        # Remove tiny YOLO jitter

        if bag_movement < 2:
            bag_movement = 0

        if person_movement < 2:
            person_movement = 0

        bag_moving = (
            bag_movement >=
            BAG_MOVEMENT_THRESHOLD
        )

        person_moving = (
            person_movement >=
            PERSON_MOVEMENT_THRESHOLD
        )

        bag_stationary = not bag_moving

        # ====================================================
        # MEDIAPIPE HANDS
        # ====================================================

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

            last_hand_points = (
                detected_hands
            )

            hand_missing = 0

        else:

            hand_missing += 1

            if hand_missing <= HAND_MEMORY_FRAMES:

                hand_points = (
                    last_hand_points
                )

            else:

                hand_points = []

        # ====================================================
        # HAND / BAG
        # ====================================================

        hand_bag_distance = (
            nearest_hand_distance(
                hand_points,
                bag_center
            )
        )

        bag_near_hand = (
            hand_bag_distance is not None
            and
            hand_bag_distance <=
            HAND_BAG_DISTANCE
        )

        gesture = get_gesture(
            hand_points
        )

        # ====================================================
        # STATE MACHINE
        # ====================================================

        old_state = state

        # ----------------------------------------------------
        # WAITING
        # ----------------------------------------------------

        if state == "WAITING":

            if bag_center is not None:

                state = "BAG DETECTED"

        # ----------------------------------------------------
        # BAG DETECTED
        # ----------------------------------------------------

        elif state == "BAG DETECTED":

            if (
                person_center is not None
                and bag_near_hand
            ):

                picking_counter += 1

            else:

                picking_counter = max(
                    0,
                    picking_counter - 1
                )

            if picking_counter >= PICKING_FRAMES:

                state = (
                    "PERSON PICKING BAG"
                )

                picking_counter = 0

        # ----------------------------------------------------
        # PICKING
        # ----------------------------------------------------

        elif state == "PERSON PICKING BAG":

            pickup_signal = (
                bag_near_hand
                and
                (
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

                state = (
                    "PERSON PICKED BAG"
                )

                was_picked = True

                picked_counter = 0

        # ----------------------------------------------------
        # PICKED
        # ----------------------------------------------------

        elif state == "PERSON PICKED BAG":

            if (
                person_center is not None
                and bag_center is not None
            ):

                carrying_counter += 1

            if carrying_counter >= CARRYING_FRAMES:

                state = (
                    "PERSON CARRYING BAG"
                )

                was_carried = True

                carrying_counter = 0

        # ----------------------------------------------------
        # CARRYING
        # ----------------------------------------------------

        elif state == "PERSON CARRYING BAG":

            if bag_stationary:

                kept_counter += 1

            else:

                kept_counter = max(
                    0,
                    kept_counter - 1
                )

            if kept_counter >= KEPT_FRAMES:

                state = (
                    "PERSON KEPT BAG"
                )

                was_kept = True

                kept_counter = 0

        # ----------------------------------------------------
        # KEPT
        # ----------------------------------------------------

        elif state == "PERSON KEPT BAG":

            if (
                person_moving
                and bag_stationary
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

        # ----------------------------------------------------
        # PLACED
        # ----------------------------------------------------

        elif state == "BAG PLACED":

            pass

        # ====================================================
        # SAVE EVENT
        # ====================================================

        if state != old_state:

            events.append({
                "frame": frame_number,
                "state": state
            })

        # ====================================================
        # DRAW DETECTIONS
        # ====================================================

        for box, confidence in persons:

            draw_box(
                frame,
                box,
                f"PERSON {confidence:.2f}",
                (0, 255, 0)
            )

        for box, confidence in bags:

            draw_box(
                frame,
                box,
                f"BAG {confidence:.2f}",
                (255, 0, 0)
            )

        # ====================================================
        # POSE
        # ====================================================

        pose_results = pose_model.predict(
            frame,
            conf=0.35,
            verbose=False
        )

        draw_pose(
            frame,
            pose_results
        )

        # ====================================================
        # HANDS
        # ====================================================

        draw_hands(
            frame,
            hand_points
        )

        # ====================================================
        # HAND → BAG LINE
        # ====================================================

        if (
            bag_center is not None
            and hand_points
        ):

            nearest_point = None
            nearest_distance_value = (
                float("inf")
            )

            for hand in hand_points:

                for point in hand:

                    d = distance(
                        point,
                        bag_center
                    )

                    if d < nearest_distance_value:

                        nearest_distance_value = d
                        nearest_point = point

            if nearest_point:

                cv2.line(
                    frame,
                    nearest_point,
                    bag_center,
                    (0, 255, 255),
                    2
                )

        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (500, 220),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            "VISIONDETECT AI",
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
            0.62,
            (0, 255, 255),
            2
        )

        distance_text = (
            "N/A"
            if hand_bag_distance is None
            else f"{hand_bag_distance:.0f}px"
        )

        cv2.putText(
            frame,
            f"Hand-Bag: {distance_text}",
            (20, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Bag Movement: {bag_movement:.1f}",
            (20, 158),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Person Movement: {person_movement:.1f}",
            (20, 186),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2
        )

        # ====================================================
        # EVENT BAR
        # ====================================================

        cv2.rectangle(
            frame,
            (10, height - 65),
            (width - 10, height - 10),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            state,
            (30, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            3
        )

        # ====================================================
        # WRITE FRAME
        # ====================================================

        out.write(frame)

        # ====================================================
        # PREVIOUS POSITIONS
        # ====================================================

        if bag_center is not None:
            previous_bag_center = bag_center

        if person_center is not None:
            previous_person_center = (
                person_center
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    out.release()
    hand_detector.close()

    return {
        "frames_processed": frame_number,
        "final_state": state,
        "person_picked": was_picked,
        "person_carried": was_carried,
        "person_kept": was_kept,
        "bag_placed": state == "BAG PLACED",
        "events": events
    }