import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break
    
    print(frame.shape)
    cv2.imshow("Video Feed", frame)
    keypress = cv2.waitKey(1) & 0xFF

    # if the user pressed "q", then stop looping
    if keypress == ord("q"):
        break

# free up memory
cap.release()
cv2.destroyAllWindow()
