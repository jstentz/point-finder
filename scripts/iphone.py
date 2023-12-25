import cv2
import numpy as np
url = "http://172.16.0.7:4747/video"
cp = cv2.VideoCapture(url)
while(True):
    camera, frame = cp.read()
    if frame is not None:
        cv2.imshow("Frame", frame)
    q = cv2.waitKey(1)
    if q==ord("q"):
        break
cp.release()
cv2.destroyAllWindows()