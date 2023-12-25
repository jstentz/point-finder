'''
This script is used to light up each light one at a time
while taking pictures with the camera.

Usage:
python capture_pixels.py <outdir>
'''

import cv2 as cv
import sys
import time
import requests
# import numpy as np

NUM_PIXELS = 500

# color in hsv 
# the color range in opencv is (0-179, 0-255, 0-255)
# should be magenta
# COLOR_HIGH = np.array([146, 255, 255])
# COLOR_LOW = np.array([143, 200, 200])

# COLOR_HIGH = np.array([179, 255, 255])
# COLOR_LOW = np.array([0, 100, 0])

if len(sys.argv) < 2:
  print('Usage: python capture_pixels.py <outdir>')
  exit(-1)

outdir = sys.argv[1]

# open the camera 
url = "http://172.16.0.7:4747/video"
cap = cv.VideoCapture(url)

# loop over all the pixels
for i in range(NUM_PIXELS):
  # light up a single light with backend
  requests.post('http://192.168.1.25:8000/', '{"light_pattern_name": "Single", "parameters": {"fps": 60, "light": %d, "color": [255, 0, 255]}}' % i, 
                headers={'Content-Type':'application/json'})
  # wait a little
  print(f'{i+1}/{NUM_PIXELS}')
  
  prev = time.time()

  while True:
    time_elapsed = time.time() - prev
    res, image = cap.read()

    if not res:
      print('ERROR: failed to read from camera')
      continue

    if time_elapsed > 2.5:
      prev = time.time()

      # save the image based on pixel value and input output folder
      cv.imwrite(f'{outdir}/img_{i}.jpg', image)
      break

cap.release()
cv.destroyAllWindows()