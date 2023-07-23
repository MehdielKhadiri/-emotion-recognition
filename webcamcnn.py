import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import h5py


# Check if GPU is available
if not tf.test.gpu_device_name():
    print('No GPU found.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


emotion_categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
train_data_directory = "C:/Users/banjo/devlloper/ltsm/archive/train"
test_data_directory = "C:/Users/banjo/devlloper/ltsm/archive/test"

def load_data(data_directory):
    data = []
    labels = []
    for category in emotion_categories:
        path = os.path.join(data_directory, category)
        class_num = emotion_categories.index(category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = load_img(img_path, target_size=(48, 48), color_mode='grayscale')
            image = img_to_array(image)
            data.append(image)
            labels.append(class_num)
    return np.array(data), np.array(labels)

X_train, y_train = load_data(train_data_directory)
X_test, y_test = load_data(test_data_directory)

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

y_train = to_categorical(y_train, num_classes=len(emotion_categories))
y_test = to_categorical(y_test, num_classes=len(emotion_categories))

# Define model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(emotion_categories), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=32), 
          steps_per_epoch=len(X_train) / 32, 
          epochs=50, 
          verbose=1, 
          callbacks=[early_stopping, mcp_save], 
          validation_data=(X_test, y_test))

model.save('webcamcnn.h5')


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