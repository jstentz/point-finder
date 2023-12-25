import cv2
import numpy as np
import sys

if len(sys.argv) < 2:
  print('USAGE: python pixel_clicker.py <imagesdir>')
  exit(-1)

imagesdir = sys.argv[1]

NUM_PIXELS = 500
locations = np.empty((NUM_PIXELS, 2))
clicked_point = None
prev_clicked_point = None

def on_mouse_click(event, x, y, flags, param):
  global clicked_point
  if event == cv2.EVENT_LBUTTONDOWN:
    clicked_point = (x, y)

for i in range(NUM_PIXELS):
  # Load the image
  image_path = f'{imagesdir}/img_{i}.jpg'  # Replace with the actual path to your image
  image = cv2.imread(image_path)
  
  if prev_clicked_point:
    image = cv2.circle(image, prev_clicked_point, 20, (0, 0, 255))

  # Display the image
  cv2.imshow("Image", image)
  cv2.setMouseCallback("Image", on_mouse_click)

  while True:
    # Wait for the user to click
    if clicked_point is not None:
      break

    # Check for the 'Esc' key to exit
    key = cv2.waitKey(1)
    if key == 27:
      break

  # Destroy the OpenCV window
  cv2.destroyAllWindows()

  # If a point was clicked, save it in a numpy array
  if clicked_point is not None:
    clicked_pixel = np.array(clicked_point)
    locations[i] = clicked_pixel
    print("Clicked Pixel:", clicked_pixel)

  prev_clicked_point = clicked_point
  clicked_point = None

# save the array 
np.save(f'{imagesdir}/locations', locations)

