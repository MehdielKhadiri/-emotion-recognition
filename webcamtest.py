import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Check if GPU is available
if not tf.test.gpu_device_name():
    print('No GPU found.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

emotion_categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load the trained model
model = load_model('webcamcnn.h5')

# Use OpenCV to capture video frames from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the grayscale image
    gray = cv2.resize(gray, (48, 48))
    
    # Reshape and normalize the grayscale image
    gray_model_input = gray.reshape(-1, 48, 48, 1) / 255.0

    # Predict the emotion of the face
    emotion = model.predict(gray_model_input)
    emotion_probability = np.max(emotion)
    emotion_label = emotion_categories[np.argmax(emotion)]

    print('Emotion: ', emotion_label)
    print('Emotion probability: ', emotion_probability)

    # Display the resulting frame and the predicted emotion
    cv2.imshow('frame', gray)  # Use the original gray image here
    

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()