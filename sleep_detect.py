import cv2
import mediapipe as mp
import time
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices for MediaPipe Face Mesh
# Left eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# EAR threshold — below this means eye is closed
EAR_THRESHOLD = 0.25
# Seconds before sleep alert triggers
SLEEP_TIME_LIMIT = 5

# Track when eyes first closed
eyes_closed_start = None
is_sleeping = False

def calculate_ear(landmarks, eye_indices, frame_w, frame_h):
    # Get eye landmark coordinates
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append((lm.x * frame_w, lm.y * frame_h))

    # EAR formula
    # Vertical distances
    v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    # Horizontal distance
    h = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    ear = (v1 + v2) / (2.0 * h)
    return ear

def detect_sleep(frame):
    global eyes_closed_start, is_sleeping

    frame_h, frame_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    result = face_mesh.process(rgb)

    rgb.flags.writeable = True

    status = "Awake"
    sleep_duration = 0

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # Calculate EAR for both eyes
        left_ear = calculate_ear(landmarks, LEFT_EYE, frame_w, frame_h)
        right_ear = calculate_ear(landmarks, RIGHT_EYE, frame_w, frame_h)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            # Eyes are closed
            if eyes_closed_start is None:
                eyes_closed_start = time.time()

            sleep_duration = time.time() - eyes_closed_start

            if sleep_duration >= SLEEP_TIME_LIMIT:
                status = "SLEEPING!"
                is_sleeping = True
            else:
                status = f"Eyes Closing... ({int(sleep_duration)}s)"
                is_sleeping = False
        else:
            # Eyes are open
            eyes_closed_start = None
            is_sleeping = False
            status = "Awake"

    else:
        # No face detected
        eyes_closed_start = None
        status = "No Face"

    return status, is_sleeping