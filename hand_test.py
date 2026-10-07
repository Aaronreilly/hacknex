import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/hand_landmarker.task"


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture("videos/bag4.mp4")

if not cap.isOpened():

    print("ERROR: Cannot open video")

    exit()


# ============================================================
# LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # BGR → RGB

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # MEDIAPIPE IMAGE
    # ========================================================

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # ========================================================
    # HAND DETECTION
    # ========================================================

    result = detector.detect(
        mp_image
    )


    # ========================================================
    # DRAW HAND LANDMARKS
    # ========================================================

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            points = []


            # Get 21 landmarks

            for landmark in hand:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                points.append(
                    (x, y)
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # Hand connections

            connections = [

                (0,1),
                (1,2),
                (2,3),
                (3,4),

                (0,5),
                (5,6),
                (6,7),
                (7,8),

                (0,9),
                (9,10),
                (10,11),
                (11,12),

                (0,13),
                (13,14),
                (14,15),
                (15,16),

                (0,17),
                (17,18),
                (18,19),
                (19,20)
            ]


            for a, b in connections:

                cv2.line(
                    frame,
                    points[a],
                    points[b],
                    (255,255,0),
                    2
                )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "MediaPipe Hand Test",
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

detector.close()

cv2.destroyAllWindows()