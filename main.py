import cv2
import pyttsx3
import torch
import speech_recognition as sr
import datetime
import webbrowser
import os
import wikipedia
import socket
import requests
import sys
import time
import signal
import mediapipe as mp
from sos import get_location, send_email  # Import SOS functions
from face_rec import recognize_faces  # Import your face recognition function

# API Configuration (Google Gemini)
API_KEY = "AIzaSyA9-d-fjcVhP5VMiDj1FtrCTYPCy8Yv0sQ"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Set speaking speed

# Load YOLOv5 model (using the lightweight version)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5s model

# Distance calculation parameters
FOCAL_LENGTH = 600  # Focal length in pixels (calibrate for your camera)
KNOWN_WIDTH = 50    # Known width of a person in cm
DISTANCE_THRESHOLD = 60  # Distance threshold in cm

# Cache to store recognized faces to avoid repeated alerts
recognized_faces_cache = set()

# Gesture tracking variables
prev_landmarks = None
gesture_text = ""
cooldown_start_time = None
cooldown_duration = 2  # Cooldown duration in seconds
listening_active = True  # To stop listening when user says "stop listening"

# Signal handler for graceful exit
def signal_handler(sig, frame):
    print("\nGesture Detection Stopped.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def announce_message(text):
    """Announce a message using text-to-speech."""
    engine.say(text)
    engine.runAndWait()

def calculate_distance(bbox_width, known_width=KNOWN_WIDTH, focal_length=FOCAL_LENGTH):
    """Calculate distance to an object using the pinhole camera model."""
    if bbox_width > 0:
        return (known_width * focal_length) / bbox_width
    return None

def process_face_recognition(frame):
    """Perform face recognition directly on the frame."""
    face_roi_path = "temp_face.jpg"
    cv2.imwrite(face_roi_path, frame)  # Save the current frame as a temporary image
    recognized_faces = recognize_faces(face_roi_path, "face_encodings.pkl")
    return recognized_faces

def process_yolo_detection(frame, last_detection_time):
    """Perform object detection using YOLOv5 on the frame."""
    results = model(frame)
    detections = results.xyxy[0]
    current_time = time.time()

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        bbox_width = x2 - x1
        label = results.names[int(cls)]

        if label == 'person':
            # Calculate distance
            distance = calculate_distance(bbox_width)
            text = f"{label} {int(distance)} cm" if distance else label

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if distance and distance < DISTANCE_THRESHOLD and current_time - last_detection_time > 5:
                last_detection_time = current_time
                announce_message(f"Person detected at {int(distance)} cm!")
                print(f"Person detected at {int(distance)} cm!")

        elif label in ['10 rupee', '20 rupee', '50 rupee', '100 rupee', '200 rupee', '500 rupee', '2000 rupee']:
            # Announce currency detection
            announce_message(f"{label} note detected.")
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame, last_detection_time

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

def perform_action(gesture):
    if gesture == "Swipe Up" or gesture == "Swipe Down" or gesture == "Swipe Left" or gesture == "Swipe Right" :
        print("Action: Activating Assistant")
        announce_message("Assistant Activated!")  # Announce activation
        run_virtual_assistant()  # Call the assistant function

def is_internet_available():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    global listening_active
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)

        for _ in range(3):  # Retry up to 3 times
            try:
                audio = recognizer.listen(source, timeout=5)  # Timeout after 5 seconds
                query = recognizer.recognize_google(audio)
                print(f"User  said: {query}")

                # Stop listening if user says "stop listening"
                if "stop listening" in query:
                    listening_active = False
                    speak("I will stop listening now.")
                    return None

                return query.lower()
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand. Trying again...")
            except sr.RequestError:
                print("Speech recognition service is unreachable. Retrying...")
                time.sleep(2)  # Wait before retrying
            except sr.WaitTimeoutError:
                print("No input detected. Please try again.")

        print("Maximum retries reached. Please type your command.")
        return None

def get_ai_response(prompt):
    if is_internet_available():
        for _ in range(3):  # Retry up to 3 times
            try:
                data = {"contents": [{"parts": [{"text": prompt}]}]}
                response = requests.post(GEMINI_URL, json=data)

                if response.status_code == 200:
                    return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "I couldn't generate a response.")
                else:
                    print(f"Error: {response.text}")
                    return "I couldn't fetch the response. Try again later."
            except requests.exceptions.RequestException:
                print("API request failed. Retrying...")
                time.sleep(2)  # Wait before retrying

        return "I am having trouble connecting to the internet."
    else:
        return "I'm currently offline. Please check your internet connection."

def start_payment():
    """Opens the payment URL (replace with actual payment link)."""
    speak("Starting the payment process.")
    webbrowser.open("https://your-payment-url.com")  # Replace with your actual payment URL

def check_balance():
    """Fetches and speaks the current balance."""
    try:
        # Make an HTTP request to fetch balance
        url = "https://kapdedholo.com/balance.php?account=ACC1001"
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the balance from the response (assuming the response is plain text or JSON)
            current_balance = response.text.strip()  # Adjust parsing logic based on actual response format
            speak(f"Your current balance is {current_balance} rupees.")
        else:
            speak("Unable to fetch your balance. Please try again later.")
    except Exception as e:
        speak(f"An error occurred while checking your balance: {e}")

def sos_alert():
    speak("Emergency detected! Sending SOS alert.")
    try:
        location = get_location()
        send_email(location)
        speak("SOS alert sent successfully.")
    except Exception as e:
        speak("Failed to send SOS alert.")
        print(f"Error: {e}")

def execute_command(command):
    if "time" in command:
        speak(f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}.")
    elif "date" in command:
        speak(f"Today's date is {datetime.datetime.today().strftime('%B %d, %Y')}.")
    elif "wikipedia" in command:
        speak("Searching Wikipedia...")
        query = command.replace("search wikipedia for", "").strip()
        try:
            result = wikipedia.summary(query, sentences=2)
            speak(result)
        except wikipedia.exceptions.DisambiguationError:
            speak("Multiple results found. Please be more specific.")
        except wikipedia.exceptions.PageError:
            speak("No results found.")
    elif "play music" in command:
        webbrowser.open("https://www.youtube.com/results?search_query=relaxing+music")
        speak("Playing music on YouTube.")
    elif "help" in command or "sos" in command:
        sos_alert()
    elif "pay" in command:
        start_payment()
    elif "check balance" in command:
        check_balance()
    elif "stop listening" in command:
        global listening_active
        listening_active = False
        speak("I will stop listening now.")
    elif "exit" in command:
        speak("Goodbye! Have a great day.")
        sys.exit()
    else:
        ai_response = get_ai_response(command)
        speak(ai_response)

def run_virtual_assistant():
    speak("Hello! How can I assist you today?")
    global listening_active
    
    failed_attempts = 0  # Counter for failed attempts

    while listening_active:
        user_command = listen()
        if user_command:
            execute_command(user_command)
            failed_attempts = 0  # Reset the counter on successful command
        else:
            failed_attempts += 1  # Increment the counter on failure
            if failed_attempts >= 3:
                speak("I have not received any input. Shutting down now.")
                print("Shutting down due to inactivity.")
                break  # Exit the loop if there are three consecutive failures

def main():
    """Main function to handle video capture and detection logic."""
    global prev_landmarks, gesture_text, cooldown_start_time

    cap = cv2.VideoCapture(0)
    last_detection_time = 0

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Gesture Detection Started! Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Step 1: Process face recognition first
            recognized_faces = process_face_recognition(frame)
            if recognized_faces:
                for face in recognized_faces:
                    if face not in recognized_faces_cache:
                        recognized_faces_cache.add(face)
                        print(f"Recognized face: {face}")
                        announce_message(f"Recognized {face}")
                continue  # Skip YOLO processing since a face is recognized

            # Step 2: Fallback to YOLO if no face or unknown face is found
            frame, last_detection_time = process_yolo_detection(frame, last_detection_time)

            # Gesture detection
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

            # Display the frame
            cv2.imshow('Face & Object Detection', frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()