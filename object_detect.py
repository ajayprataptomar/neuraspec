import RPi.GPIO as GPIO
import time
import cv2
from ultralytics import YOLO

# Set GPIO mode
GPIO.setmode(GPIO.BCM)

# Define GPIO pins
TRIG = 18
ECHO = 25

# Set up the GPIO pins
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Load the YOLO model
model = YOLO("yolo11n.pt")

def measure_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    return round(distance, 2)

try:
    while True:
        dist = measure_distance()
        print(f"Distance: {dist} cm")

        # Capture an image from the camera
        frame = cv2.VideoCapture(0).read()[1]

        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(1)

except KeyboardInterrupt:
    print("Measurement stopped by user")
finally:
    GPIO.cleanup()
    cv2.destroyAllWindows()
    