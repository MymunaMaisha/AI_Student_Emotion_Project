import cv2
print("cv2 loaded")
from face import detect_face
print("face loaded")
from emotion import detect_emotion
print("emotion loaded")
from posture import detect_posture
print("posture loaded")
from sleep_detect import detect_sleep
print("sleep_detect loaded")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam. Trying camera index 1...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("ERROR: No webcam found.")
    exit()

print("Webcam opened! Press ESC to quit.")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Failed to grab frame")
        break

    frame = cv2.resize(frame, (640, 480))

    # --- Sleep Detection ---
    sleep_status, is_sleeping = detect_sleep(frame)

    if is_sleeping:
        cv2.putText(frame, "SLEEPING ALERT!", (80, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        cv2.rectangle(frame, (0,0),
                      (frame.shape[1], frame.shape[0]), (0,0,255), 10)
    else:
        cv2.putText(frame, f"Sleep: {sleep_status}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # --- Emotion Detection ---
    faces = detect_face(frame)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        emotion = detect_emotion(face)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # --- Posture Detection ---
    posture = detect_posture(frame)
    cv2.putText(frame, f"Posture: {posture}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.imshow("AI Student Emotion Monitor", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()