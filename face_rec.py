import face_recognition
import os
import cv2
import pickle


def recognize_faces(image_path, encoding_file, tolerance=0.4):
    """
    Recognize faces in an image and return their names.
    """
    # Load the trained encodings
    with open(encoding_file, "rb") as f:
        data = pickle.load(f)

    # Load the image and convert it to RGB
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings in the image
    boxes = face_recognition.face_locations(rgb_image)
    encodings = face_recognition.face_encodings(rgb_image, boxes)

    # List to store the names of detected faces
    names = []

    for encoding in encodings:
        # Compare the encoding with known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance)
        name = "Unknown"

        # Use the distance to find the best match
        if True in matches:
            matched_indexes = [i for i, match in enumerate(matches) if match]
            face_distances = face_recognition.face_distance(data["encodings"], encoding)
            best_match_index = matched_indexes[face_distances[matched_indexes].argmin()]
            name = data["names"][best_match_index]

        names.append(name)

    return names


def train_faces(dataset_path, encoding_file):
    print("[INFO] Processing images...")
    known_encodings = []
    known_names = []

    # Loop through all images in the dataset folder
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(("jpg", "jpeg", "png")):
                # Extract the person's name from the folder name
                name = os.path.basename(root)

                # Load the image
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Detect face encodings
                boxes = face_recognition.face_locations(rgb_image)
                encodings = face_recognition.face_encodings(rgb_image, boxes)

                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(name)

    # Save the encodings and names to a file
    data = {"encodings": known_encodings, "names": known_names}
    with open(encoding_file, "wb") as f:
        pickle.dump(data, f)

    print(f"[INFO] Encodings saved to {encoding_file}")


def recognize_faces_in_video(encoding_file, tolerance=0.4):
    """
    Recognize faces in a real-time video stream and display their names.
    """
    # Load known face encodings
    print("[INFO] Loading encodings...")
    with open(encoding_file, "rb") as f:
        data = pickle.load(f)

    # Start video capture
    print("[INFO] Starting video stream...")
    video_capture = cv2.VideoCapture(0)  # 0 for default webcam

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Unable to access webcam.")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces and compute encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance)
            name = "Unknown"

            if True in matches:
                face_distances = face_recognition.face_distance(data["encodings"], encoding)
                best_match_index = face_distances.argmin()
                name = data["names"][best_match_index]

            face_names.append(name)

        # Display results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations to the original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Display the name below the rectangle
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)


        # Show the video frame
        cv2.imshow("Video", frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release video capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Train the model
    dataset_path = "/home/rahul/Desktop/neurospec/dataset"  # Folder containing subfolders for each person
    encoding_file = "/home/rahul/Desktop/neurospec/face_encodings.pkl"
    train_faces(dataset_path, encoding_file)

    # Recognize faces in a test image
    test_image_path = "/home/rahul/Desktop/neurospec/dataset/ajay/ajay1.jpg"  # Replace with the path to your test image
    detected_names = recognize_faces(test_image_path, encoding_file)
    print("Detected names:", detected_names)

    # Recognize faces in real-time video
    recognize_faces_in_video(encoding_file)
