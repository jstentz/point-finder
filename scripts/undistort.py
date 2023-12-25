import sys
import os
import cv2 as cv
import pickle

if len(sys.argv) < 3:
  print('USAGE: python undistort.py <cameradir> <imagesdir>')

NUM_PIXELS = 500

cameradir, imagesdir = sys.argv[1], sys.argv[2]

outdir = f'{imagesdir}_undist2'

# create the undistorted directory if it does not exist 
if not os.path.exists(outdir):
  os.makedirs(outdir)

# load in the camera properties 
with open(f'{cameradir}/calibration.pkl', 'rb') as f:
  cameraMatrix, dist = pickle.load(f)

img = cv.imread(f'{imagesdir}/img_{0}.jpg')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

for i in range(NUM_PIXELS):
  img = cv.imread(f'{imagesdir}/img_{i}.jpg')

  # Undistort
  dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

  # crop the image
  x, y, w, h = roi
  dst = dst[y:y+h, x:x+w]
  cv.imwrite(f'{outdir}/img_{i}.jpg', dst)