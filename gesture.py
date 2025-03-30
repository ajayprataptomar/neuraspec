import cv2
import mediapipe as mp
import time
import signal
import sys

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture tracking variables
prev_landmarks = None
gesture_text = ""
cooldown_start_time = None
cooldown_duration = 2  # Cooldown duration in seconds

# Signal handler for graceful exit
def signal_handler(sig, frame):
    print("\nGesture Detection Stopped.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Detect gestures based on motion
def detect_motion_gesture(current_landmarks, prev_landmarks):
    """Detects swipe gestures based on motion between frames."""
    global gesture_text

    if prev_landmarks is None:
        return None  # No previous landmarks to compare

    # Calculate horizontal and vertical movement
    curr_index_x = current_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    curr_index_y = current_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    prev_index_x = prev_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    prev_index_y = prev_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

    dx = curr_index_x - prev_index_x
    dy = curr_index_y - prev_index_y

    # Detect gestures based on movement
    if abs(dx) > abs(dy):  # Horizontal motion
        if dx > 0.05:  # Swipe Right
            gesture_text = "Swipe Right"
            return "Swipe Right"
        elif dx < -0.05:  # Swipe Left
            gesture_text = "Swipe Left"
            return "Swipe Left"
    else:  # Vertical motion
        if dy > 0.05:  # Swipe Down
            gesture_text = "Swipe Down"
            return "Swipe Down"
        elif dy < -0.05:  # Swipe Up
            gesture_text = "Swipe Up"
            return "Swipe Up"

    return None  # No significant movement detected

# Perform action based on gesture
def perform_action(gesture):
    if gesture == "Swipe Up":
        print("Action: Scroll Up")
    elif gesture == "Swipe Down":
        print("Action: Scroll Down")
    elif gesture == "Swipe Left":
        print("Action: Go Back")
    elif gesture == "Swipe Right":
        print("Action: Next Item")

# Main function
def main():
    global prev_landmarks, gesture_text, cooldown_start_time

    cap = cv2.VideoCapture(0)
    print("Gesture Detection Started! Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Check for hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect gesture based on motion
                gesture = detect_motion_gesture(hand_landmarks.landmark, prev_landmarks)

                # Update previous landmarks
                prev_landmarks = hand_landmarks.landmark

                # Perform action if a gesture is detected
                if gesture:
                    current_time = time.time()
                    if cooldown_start_time is None or (current_time - cooldown_start_time > cooldown_duration):
                        print(f"Gesture Detected: {gesture}")
                        perform_action(gesture)
                        cooldown_start_time = current_time

        # Display the detected gesture on the video feed
        if gesture_text:
            cv2.putText(frame, f"Gesture: {gesture_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Gesture Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the program
if __name__ == "__main__":
    main()
