import cv2
import os

# Set up the camera
cap = cv2.VideoCapture(0)

# Set up the folder structure
gesture_classes = ['fist', 'palm', 'v_sign', 'thumbs_up', 'ok_sign']
for gesture in gesture_classes:
    os.makedirs(gesture, exist_ok=True)

# Capture images
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame
    cv2.imshow('Hand Gesture', frame)

    # Capture an image on pressing 'c'
    if cv2.waitKey(1) & 0xFF == ord('c'):
        gesture = input("Enter the gesture class: ")
        cv2.imwrite(os.path.join(gesture, f"{gesture}_{len(os.listdir(gesture))}.jpg"), frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the hand gesture classes
gesture_classes = ['fist', 'palm', 'v_sign', 'thumbs_up', 'ok_sign']

# Load the dataset
train_dir = 'hand_gestures/train/'
test_dir = 'hand_gestures/test/'

train_images = []
train_labels = []
for gesture in gesture_classes:
    for file in os.listdir(train_dir + gesture):
        img = cv2.imread(train_dir + gesture + '/' + file)
        img = cv2.resize(img, (224, 224))
        train_images.append(img)
        train_labels.append(gesture_classes.index(gesture))

test_images = []
test_labels = []
for gesture in gesture_classes:
    for file in os.listdir(test_dir + gesture):
        img = cv2.imread(test_dir + gesture + '/' + file)
        img = cv2.resize(img, (224, 224))
        test_images.append(img)
        test_labels.append(gesture_classes.index(gesture))

# Convert the lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Normalize the pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(gesture_classes), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

# Evaluate the model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {accuracy:.2f}')

# Use the model for real-time hand gesture recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)

    # Make predictions
    predictions = model.predict(frame)
    predicted_class = np.argmax(predictions)

    # Draw the predicted gesture on the frame
    cv2.putText(frame, gesture_classes[predicted_class], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()