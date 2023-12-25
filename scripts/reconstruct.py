'''
Read in the points and do the triangulation.
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle

# read in the points
pts1 = np.load('../images/frontleft_undist/locations.npy')
pts2 = np.load('../images/frontright_undist/locations.npy')

# filter out the points that are not visible from both angles
filled1 = np.any(pts1 != np.array([-1, -1]), axis=1)
filled2 = np.any(pts2 != np.array([-1, -1]), axis=1)

both_filled = filled1 & filled2

pts1 = pts1[both_filled]
pts2 = pts2[both_filled]

# convert to floating points (not sure if this matters)
pts1 = pts1.astype(np.float64)
pts2 = pts2.astype(np.float64)

# compute the fundamental matrix between the two frames
F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

# load in the camera intrinsics
K = pickle.load(open('../camera/cameraMatrix.pkl', 'rb'))

print(K)

# compute the essential matrix
E = K.T @ F @ K # TODO: should I normalize this by E[-1, -1]?
E /= E[-1, -1]

# compute the extrinsics for camera 1 (use global == camera 1)
identity_extrinsics = np.array([
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0]
])

P1 = K @ identity_extrinsics

# compute the possibilities for the extrinsic parameters for camera 1

# function stolen from 16-385 assgn3 helper file
def camera2(E):
  U,S,V = np.linalg.svd(E)
  m = S[:2].mean()
  E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
  U,S,V = np.linalg.svd(E)
  W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

  if np.linalg.det(U.dot(W).dot(V))<0:
      W = -W

  M2s = np.zeros([3,4,4])
  M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
  M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
  M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
  M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
  return M2s

extrinsic_options = camera2(E)

# use these possibilities to reconstruct the points 4 times
pts3 = np.empty((4, pts1.shape[0], 3))
for i in range(extrinsic_options.shape[2]):
  P2 = K @ extrinsic_options[:, :, i]
  # pts = triangulate(P1, pts1, P2, pts2)
  pts = cv.triangulatePoints(P1, P2, pts1.T, pts2.T).T

  # divide by final coordinate (homogeneous) and remove the final coord
  pts3[i] = (pts / np.expand_dims(pts[:, -1], -1))[:, :3]

# find the extrinsics that put the points in front of the camera
onlyz = pts3[:, :, 2]
positive_z_sums = np.sum(onlyz >= 0, axis=1)
best_idx = np.argmax(positive_z_sums)
final_pts = pts3[best_idx]

# swap them so z is the last column
final_pts[:, [2, 1]] = final_pts[:, [1, 2]]

# negate the z axis
final_pts[:, 2] *= -1

# center the points around the mean and make max distance be sqrt 2
mean_pt = np.mean(final_pts, axis=0)
final_pts -= mean_pt # center around the mean
max_dist = np.max(np.linalg.norm(final_pts, axis=1))
final_pts = (final_pts / max_dist) * np.sqrt(2)


# display the points

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.scatter(final_pts[:, 0], final_pts[:, 1], final_pts[:, 2])
ax.set_aspect('equal')
plt.show()


'''
Possible sources of scaling error:
1. Not good enough of a camera matrix, might need to make sure the calibration was done correctly...
This would suck if I had to undistort the images again and reclick on the things

2. something wrong with my math (unlikely cuz I did it)

3. not resolute enough (I don't think this is the problem)

4. I am not passing a wide enough range of points into findFundamentalMatrix. I could consider
clicking on other parts of the scene that correspond and see if that helps. Lowering the number of points
passed into that function seems to only worsen things in terms of scaling.

If the scaling is off, I'm worried about bringing in data from other places, since there might 
be even more error created by that

in the parameters, there is also "size of chessboard in mm", which I didn't take the time to measure
out when I put it on the screen... maybe try changing that?

'''