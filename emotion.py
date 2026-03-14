import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model.h5", compile=False)

labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def detect_emotion(face):
    face = cv2.resize(face, (64,64))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face / 255.0
    face = np.reshape(face, (1,64,64,1))
    prediction = model.predict(face, verbose=0)
    return labels[prediction.argmax()]