import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import random

# Load model and class names
model = load_model('mp_hand_gesture')
classNames = open('gesture.names', 'r').read().split('\n')

# Specify the desired gestures directly in the code
desired_gestures = ["thumbs up", "thumbs down", "peace", "okay", "stop"]  # Replace with your desired gestures

# Filter the class names to include only the desired gestures
filtered_classNames = [gesture for gesture in classNames if gesture.lower() in [g.lower() for g in desired_gestures]]

# Initialize MediaPipe and webcam
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Define game parameters
total_game_time = 30  # Total time for the entire game in seconds

def generate_gesture_sequence(length):
    """Generates a random sequence of distinct gestures from the filtered list."""
    return random.sample(filtered_classNames, length)

def start_game():
    """Begins the game loop, tracking player gestures and scores."""
    gesture_sequence = generate_gesture_sequence(5)  # Generate a sequence of 5 gestures
    current_gesture_index = 0
    start_time = time.time()
    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(framergb)
        className = ''

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * frame.shape[1])
                    lmy = int(lm.y * frame.shape[0])
                    landmarks.append([lmx, lmy])

                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                className = classNames[classID]

            if className == gesture_sequence[current_gesture_index]:
                current_gesture_index += 1
                cv2.putText(frame, "Correct!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check for game completion or timeout
            if current_gesture_index == len(gesture_sequence) or time.time() - start_time > total_game_time:
                cv2.putText(frame, "Game Over! Score: " + str(current_gesture_index), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break

        cv2.putText(frame, "Target: " + gesture_sequence[current_gesture_index], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Time Remaining: " + str(int(total_game_time - (time.time() - start_time))), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Gesture Game", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start a single game
start_game()