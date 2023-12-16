import cv2 as cv
import numpy as np

# cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap = cv.VideoCapture(-1)

# Check if the webcam is opened correctly
if not cap.isOpened():
  raise IOError("Cannot open webcam")

while True:
  ret, frame = cap.read()
  frame = cv.flip(frame, 1)
  frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

  ret, corners = cv.findChessboardCorners(frame, (6,6), None)
  
  if ret:
    cv.drawChessboardCorners(frame, (6,6), corners, ret)

  cv.imshow('Corners', frame)

  c = cv.waitKey(1)
  if c == 27:
    break

cap.release()
cv.destroyAllWindows()