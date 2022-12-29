import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lowerRed = np.array([150,150,100])
    upperRed = np.array([255,255,255])

    mask = cv.inRange(hsv, lowerRed, upperRed)

    res = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow('Input', res)

    c = cv.waitKey(1)
    if c == 27:
        break

cap.release()
cv.destroyAllWindows()