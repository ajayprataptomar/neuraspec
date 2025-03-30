
from face_rec import recognize_faces

if __name__ == "__main__":
    encoding_file = "/home/rahul/Desktop/neurospec/face_encodings.pkl"
   # Recognize faces in a test image
    test_image_path = "/home/rahul/Desktop/neurospec/dataset/ajay/ajay1.jpg"  # Replace with the path to your test image
    detected_names = recognize_faces(test_image_path, encoding_file)
    print("Detected names:", detected_names)
