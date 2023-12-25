import cv2 as cv
import numpy as np
import sys

if len(sys.argv) < 2:
  print('USAGE: python test.py <backleft/backright/frontleft/frontright>')
  exit(-1)

angle = sys.argv[1]
imagesdir = f'../images/{angle}_undist2'
points_file = f'../images/{angle}_undist/locations.npy'

locations = np.load(points_file)

NUM_PIXELS = 500

for i in range(NUM_PIXELS):
  loc = locations[i]
  if tuple(loc) == (-1, -1):
    continue

  image_path = f'{imagesdir}/img_{i}.jpg'
  image = cv.imread(image_path)
  assert image is not None

  image = cv.circle(image, loc, 3, (0, 0, 255))

  cv.imshow("Image", image)

  while True:
    # 8 is backspace (go back to the previous image)
    # 32 is space (i don't know where point is)
    # 13 is enter (select where my mouse is)
    # Check for the 'Esc' key to exit
    key = cv.waitKey(10)
    if key == 32: # space (I don't know where it is)
      break
cv.destroyAllWindows()