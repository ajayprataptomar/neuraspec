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
from face_rec import recognize_faces  # Import your face recognition function
from sos import get_location, send_email  # Import SOS functions
import threading

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Set speaking speed
engine.setProperty('volume', 1.0)  # Volume

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5s model

# Distance calculation parameters
FOCAL_LENGTH = 600  # Focal length in pixels (calibrate for your camera)
KNOWN_WIDTH = 50    # Known width of a person in cm
DISTANCE_THRESHOLD = 60  # Distance threshold in cm

# Cache to store recognized faces to avoid repeated alerts
recognized_faces_cache = set()

# Global variable to control listening state
listening_active = True

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
                if listening_active:  # Only announce if not listening
                    announce_message(f"Person detected at {int(distance)} cm!")
                print(f"Person detected at {int(distance)} cm!")

        elif label in ['10 rupee', '20 rupee', '50 rupee', '100 rupee', '200 rupee', '500 rupee', '2000 rupee']:
            # Announce currency detection
            if listening_active:  # Only announce if not listening
                announce_message(f"{label} note detected.")
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame, last_detection_time

def face_and_object_recognition():
    """Run face and object recognition in a separate thread."""
    cap = cv2.VideoCapture(0)
    last_detection_time = 0

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

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
                        if listening_active:  # Only announce if not listening
                            announce_message(f"Recognized {face}")
                continue  # Skip YOLO processing since a face is recognized

            # Step 2: Fallback to YOLO if no face or unknown face is found
            frame, last_detection_time = process_yolo_detection(frame, last_detection_time)

            # Display the frame
            cv2.imshow('Face & Object Detection', frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

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

                # Check for activation keyword
                if "hello hello" in query.lower():
                    # Proceed to listen for commands after activation keyword
                    return query.lower()
                else:
                    print("Activation keyword not detected.")
                    continue  # Continue listening

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
    
    while True:
        user_command = listen()
        if user_command:
            execute_command(user_command)

if __name__ == "__main__":
    # Start the face and object recognition in a separate thread
    recognition_thread = threading.Thread(target=face_and_object_recognition)
    recognition_thread.start()

    # Run the virtual assistant
    run_virtual_assistant()