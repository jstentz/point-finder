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

frontleft_points = np.load('corresp_frontleft.npy')
frontright_points = np.load('corresp_frontright.npy')

F, mask = cv.findFundamentalMat(frontleft_points, frontright_points, cv.FM_LMEDS)

# load in a random image
I1 = cv.cvtColor(cv.imread('../images/frontleft_undist/img_0.jpg'), cv.COLOR_BGR2RGB)
I2 = cv.cvtColor(cv.imread('../images/frontright_undist/img_0.jpg'), cv.COLOR_BGR2RGB)

displayEpipolarF(I1, I2, F)