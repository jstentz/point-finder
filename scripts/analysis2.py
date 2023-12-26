import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# run the eight point algo to get E?

def _epipoles(E):
  U, S, V = np.linalg.svd(E)
  e1 = V[-1, :]
  U, S, V = np.linalg.svd(E.T)
  e2 = V[-1, :]

  return e1, e2

def displayEpipolarF(I1, I2, F):
  e1, e2 = _epipoles(F)

  sy, sx, _ = I2.shape

  f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
  ax1.imshow(I1)
  ax1.set_title('Select a point in this image')
  ax1.set_axis_off()
  ax2.imshow(I2)
  ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
  ax2.set_axis_off()

  while True:
    plt.sca(ax1)
    x, y = plt.ginput(1, mouse_stop=2)[0]

    xc, yc = int(x), int(y)
    v = np.array([[xc], [yc], [1]])

    l = F @ v
    s = np.sqrt(l[0]**2+l[1]**2)

    if s==0:
      print('Zero line vector in displayEpipolar')

    l = l / s
    if l[1] != 0:
      xs = 0
      xe = sx - 1
      ys = -(l[0] * xs + l[2]) / l[1]
      ye = -(l[0] * xe + l[2]) / l[1]
    else:
      ys = 0
      ye = sy - 1
      xs = -(l[1] * ys + l[2]) / l[0]
      xe = -(l[1] * ye + l[2]) / l[0]

    ax1.plot(x, y, '*', markersize=6, linewidth=2)
    ax2.plot([xs, xe], [ys, ye], linewidth=2)
    plt.draw()

# read in the points
pts1 = np.load('../images/frontleft_undist2/locations.npy').astype(np.float64)
pts2 = np.load('../images/frontright_undist2/locations.npy').astype(np.float64)
pts3 = np.load('../images/backleft_undist2/locations.npy').astype(np.float64)

# filter out the points that are not visible from both angles
filled1 = np.any(pts1 != np.array([-1, -1]), axis=1)
filled2 = np.any(pts2 != np.array([-1, -1]), axis=1)
filled3 = np.any(pts3 != np.array([-1, -1]), axis=1)

FL_FR_shared = filled1 & filled2
FL_BR_shared = filled1 & filled3

# extract the relevant points shared between angles
shared_pts1 = pts1[FL_FR_shared]
shared_pts2 = pts2[FL_FR_shared]

shared_pts3 = pts1[FL_BR_shared]
shared_pts4 = pts3[FL_BR_shared]

F, mask = cv.findFundamentalMat(shared_pts3, shared_pts4, cv.FM_8POINT)

# load in a random image
I1 = cv.cvtColor(cv.imread('../images/frontleft_undist2/img_0.jpg'), cv.COLOR_BGR2RGB)
I2 = cv.cvtColor(cv.imread('../images/backleft_undist2/img_0.jpg'), cv.COLOR_BGR2RGB)

displayEpipolarF(I1, I2, F)