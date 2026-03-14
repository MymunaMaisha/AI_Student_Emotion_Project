import cv2
from face import detect_face
from emotion import detect_emotion
from posture import detect_posture

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam. Trying camera index 1...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("ERROR: No webcam found. Please check your camera.")
    exit()

print("Webcam opened successfully! Press ESC to quit.")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Failed to grab frame")
        break

    faces = detect_face(frame)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        emotion = detect_emotion(face)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    posture = detect_posture(frame)
    cv2.putText(frame, posture, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("AI Student Emotion Monitor", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()