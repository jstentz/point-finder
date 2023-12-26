'''
This script is for easy hand labeling of correspondences.

USAGE: python hand_label.py <backleft/backright/frontleft/frontright>
'''

import cv2 as cv
import numpy as np
import sys

if len(sys.argv) < 2:
  print('USAGE: python hand_label.py <backleft/backright/frontleft/frontright>')
  exit(-1)

angle = sys.argv[1]
imagesdir = f'../images/{angle}_undist2'

NUM_PIXELS = 500

# start with all -1 for now (which means we don't know any locations)
# locations = (np.ones((NUM_PIXELS, 2)) * -1).astype(np.int64)
locations = np.load(f'../images/{angle}_undist2/guess_locations.npy').astype(np.int64)
mouse_loc = None

def on_mouse_click(event, x, y, flags, param):
  global mouse_loc
  if event == cv.EVENT_MOUSEMOVE:
    mouse_loc = (x, y)

i = 0
while i < NUM_PIXELS:
  print(f'Looking at light {i + 1}/{NUM_PIXELS}')
  # Load the image
  image_path = f'{imagesdir}/img_{i}.jpg'
  image = cv.imread(image_path)

  assert image is not None
  
  # draw the area where we last clicked if we know where that point is
  if i != 0 and tuple(locations[i-1]) != (-1, -1):
    image = cv.circle(image, tuple(locations[i-1]), 20, (0, 0, 255))

  # Display the image
  cv.imshow("Image", image)
  cv.setMouseCallback("Image", on_mouse_click)

  while True:
    # 8 is backspace (go back to the previous image)
    # 32 is space (i don't know where point is)
    # 13 is enter (select where my mouse is)
    # Check for the 'Esc' key to exit
    key = cv.waitKey(1)
    if key == 115: # s
      print('Going back!')
      i = max(0, i - 1)
      break
    elif key == 119: # w
      print('Going forward!')
      i = i + 1
      break
    elif key == 13: # enter (I don't know where it is)
      print('Recorded light not visible')
      locations[i] = np.array([-1, -1])
      i += 1
      break
    elif key == 32: # space (pick where my mouse currently is) 
      print(f'Selecting {mouse_loc}')
      locations[i] = np.array(mouse_loc).astype(np.int64)
      i += 1
      break

# save the array 
np.save(f'{imagesdir}/guess_locations.npy', locations)