'''
Goal of this script is to automatically find the pixel light locations of
all the pixels that are visible, which has been hand labeled by me.

The goal would be to hopefully also be able to tell if the pixels are not there
and label that data point accordingly.

New idea: hand label enough points in each of the images to be able to get the essential matrix
between any two pairs of 90 degree camera angles. (maybe 20 or 30 correspondences, we will see what works best)
then I can limit the search space to the line from prev image (with a threshold), alongside a circle from the location
of the previous LED location.

^^^ never mind this doesn't really work because that implies we have the other location
let me first just try this method and then identify the known led locations


Here is a thread about someone possibly trying to solve a similar problem?
https://forum.opencv.org/t/how-to-detect-a-led-light-source-on-an-image/1428/6

use grayscale, threshold, findContours, and minEnclosingCircle


import numpy as np
import cv2 as cv
im = cv.imread('test.jpg')
assert im is not None, "file could not be read, check with os.path.exists()"
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


'''

import cv2 as cv
import numpy as np

# cv.findContours()

# test image for now
img = cv.imread('../images/backleft/img_6.jpg')

# img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# lower_blue = np.array([110,150,150]) 
# upper_blue = np.array([180,255,255]) 

# # Here we are defining range of bluecolor in HSV 
# # This creates a mask of blue coloured  
# # objects found in the frame. 
# mask = cv.inRange(img, lower_blue, upper_blue) 

# # The bitwise and of the frame and mask is done so  
# # that only the blue coloured objects are highlighted  
# # and stored in res 
# img = cv.bitwise_and(img, img, mask=mask) 

# img = cv.cvtColor(img, cv.COLOR_HSV2BGR)

imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 200, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
center, radius = cv.minEnclosingCircle(contours)
imgray = cv.circle(imgray,center,radius,(0,255,0),2)
imgray = cv.drawContours(imgray, contours, -1, (0,255,0), 3)


# display the image
cv.imshow('Image', imgray)

while True:
  k = cv.waitKey(1)
  if k != -1:
    break

cv.destroyAllWindows()



'''
Notes:
 * it might be good to filter out by HSV
 * we want pixels that are both bright and blue
 * can consider reducing search space to the line using epipolar correspondences, but
 we would need to know that pixel's location in the other image 
'''