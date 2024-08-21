Overview
This project implements a hand gesture recognition system using a Convolutional Neural Network (CNN) model and OpenCV. The system is capable of classifying various hand gestures in real-time, including gestures like a fist, palm, v-sign, thumbs up, and ok-sign.

The project is divided into two parts:

Capturing hand gesture images for building the dataset.
Training a CNN model for gesture recognition and using it for real-time predictions.
Requirements
To run this project, you'll need the following Python packages:

opencv-python
numpy
tensorflow
You can install the required packages using pip:

bash
Copy code
pip install opencv-python numpy tensorflow
Project Structure
The project consists of the following main components:

Dataset Creation:

Captures images of different hand gestures using your webcam.
Saves the images in respective folders for each gesture class.
CNN Model Training:

Loads the hand gesture images and trains a CNN model on them.
The model is trained to recognize the gestures and tested on a separate test set.
Real-Time Hand Gesture Recognition:

Uses the trained model to recognize hand gestures in real-time through the webcam.
Dataset Structure
The dataset should be organized in the following folder structure:

bash
Copy code
hand_gestures/
│
├── train/
│   ├── fist/
│   ├── palm/
│   ├── v_sign/
│   ├── thumbs_up/
│   └── ok_sign/
│
└── test/
    ├── fist/
    ├── palm/
    ├── v_sign/
    ├── thumbs_up/
    └── ok_sign/
Code Explanation
1. Dataset Creation
The first part of the project captures images for each gesture class using your webcam:

python
Copy code
# Set up the camera
cap = cv2.VideoCapture(0)

# Capture images for each gesture class
while True:
    ret, frame = cap.read()
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
2. CNN Model Training
Once you have collected enough data, the second part of the code loads the dataset and trains a CNN model on it:

python
Copy code
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

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))
3. Real-Time Gesture Recognition
After training, you can use the trained model for real-time gesture recognition:

python
Copy code
# Use the model for real-time hand gesture recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)

    # Make predictions
    predictions = model.predict(frame)
    predicted_class = np.argmax(predictions)

    # Display the output
    cv2.putText(frame, gesture_classes[predicted_class], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
Usage
Capture Images for Dataset:

Run the first part of the code to capture images of different hand gestures using your webcam.
Press 'c' to capture an image and specify the gesture class (e.g., fist, palm).
Images will be saved in their respective folders.
Train the CNN Model:

After collecting enough data, run the second part of the code to train the CNN model.
The model will be trained on the collected data and evaluated on the test set.
Real-Time Gesture Recognition:

Once the model is trained, run the third part of the code to use the model for real-time hand gesture recognition.
The webcam feed will display the recognized gesture on the screen.
Conclusion
This project provides a basic framework for hand gesture recognition using a CNN model and OpenCV. You can further enhance the project by adding more gesture classes, improving the dataset, or fine-tuning the model.
