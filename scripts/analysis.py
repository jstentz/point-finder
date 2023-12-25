import cv2 as cv
import numpy as np
import pickle

backleft = np.load('../images/backleft_undist/visibility.npy')
backright = np.load('../images/backright_undist/visibility.npy')
frontleft = np.load('../images/frontleft_undist/visibility.npy')
frontright = np.load('../images/frontright_undist/visibility.npy')

print(np.sum(backleft & backright | backleft & frontleft | backright & frontright | frontleft & frontright)) # missing around 20

print(np.sum(backleft & backright))
print(np.sum(backleft & frontleft))
print(np.sum(backright & frontright))
print(np.sum(frontleft & frontright)) # this has the most overlap 


# let's pick 8 random points that overlap and find their image coordinates

num_corres = 20

true_indices = np.where(frontleft & frontright)[0]
points = true_indices[np.random.choice(true_indices.shape[0], num_corres, replace=False)]

clicked_point = None

def on_mouse_click(event, x, y, flags, param):
  global clicked_point
  if event == cv.EVENT_LBUTTONDOWN:
    clicked_point = np.array([x, y])

frontleft_points = np.empty((num_corres, 2))
frontright_points = np.empty((num_corres, 2))

for i, point in np.ndenumerate(points):
  # load image 1
  image_path = f'../images/frontleft_undist/img_{point}.jpg'
  image = cv.imread(image_path)

  # have someone click on the location
  cv.imshow("Image", image)
  cv.setMouseCallback("Image", on_mouse_click)

  while True:
    # Wait for the user to click
    if clicked_point is not None:
      break

    # Check for the 'Esc' key to exit
    key = cv.waitKey(1)
    if key == 27:
      break

  frontleft_points[i] = clicked_point
  clicked_point = None

  # load image 2 
  image_path = f'../images/frontright_undist/img_{point}.jpg'
  image = cv.imread(image_path)

  # have someone click on the location
  cv.imshow("Image", image)
  cv.setMouseCallback("Image", on_mouse_click)

  while True:
    # Wait for the user to click
    if clicked_point is not None:
      break

    # Check for the 'Esc' key to exit
    key = cv.waitKey(1)
    if key == 27:
      break

  frontright_points[i] = clicked_point
  clicked_point = None

np.save('corresp_frontleft', frontleft_points)
np.save('corresp_frontright', frontright_points)