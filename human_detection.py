from ultralytics import YOLO
import cv2
import numpy as np
import urllib.request

model = YOLO("yolov8n.pt")

url = "http://192.168.1.65:8080/shot.jpg"

while True:
    # Read image from URL
    resp = urllib.request.urlopen(url)
    #Change image to numpy array.
    image_source = np.asarray(bytearray(resp.read()), dtype="uint8")
    image_cv = cv2.imdecode(image_source, -1)
    
    # Resize image
    final_img = cv2.resize(image_cv, (600, 400))
    
    # Predict using YOLO model
    model.predict(source=final_img, show=True, save=True, conf=0.5, classes=[0])

    # Check for key press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
