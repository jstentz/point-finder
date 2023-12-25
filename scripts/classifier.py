'''
This script takes in a directory with images in it, and 
allows the user to classify whether or not they can clearly see the exact
location of the light.
'''

import cv2 as cv
import numpy as np
import sys

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('USAGE: python classifier.py <imagesdir>')

  imagesdir = sys.argv[1]

  NUM_PIXELS = 500
  visibility = np.empty(NUM_PIXELS, dtype=bool)

  for i in range(NUM_PIXELS):
    # load in the image
    image = cv.imread(f'{imagesdir}/img_{i}.jpg')

    # display the image
    cv.imshow('Image', image)

    # wait for user input (pick yes or no)
    # in this case, yes is space and no is backspace
    while True:
      key = cv.waitKey(0)
      if key == 8:  # Backspace
        visibility[i] = False
        break
      elif key == 32: # space
        # record yes
        visibility[i] = True
        break
  
  # save the resulting visibility array
  np.save(f'{imagesdir}/visibility', visibility)

  cv.destroyAllWindows()