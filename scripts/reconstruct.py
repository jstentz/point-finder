'''
Read in the points and do the triangulation.
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle

NUM_PIXELS = 500

# read in the points
pts1 = np.load('../images/frontleft_undist2/guess_locations.npy').astype(np.float64)
pts2 = np.load('../images/frontright_undist2/guess_locations.npy').astype(np.float64)
# pts3 = np.load('../images/backleft_undist2/locations.npy').astype(np.float64)

# filter out the points that are not visible from both angles
filled1 = np.any(pts1 != np.array([-1, -1]), axis=1)
filled2 = np.any(pts2 != np.array([-1, -1]), axis=1)
# filled3 = np.any(pts3 != np.array([-1, -1]), axis=1)

FL_FR_shared = filled1 & filled2
# FL_BR_shared = filled1 & filled3

# extract the relevant points shared between angles
shared_pts1 = pts1[FL_FR_shared]
shared_pts2 = pts2[FL_FR_shared]

# shared_pts3 = pts1[FL_BR_shared]
# shared_pts4 = pts3[FL_BR_shared]

# compute the fundamental matrix between frontleft and frontright
F_FR_FL, _ = cv.findFundamentalMat(shared_pts1, shared_pts2, cv.FM_8POINT)

# compute the fundamental matrix between frontleft and backleft
# F_FL_BL, _ = cv.findFundamentalMat(shared_pts3, shared_pts4, cv.FM_8POINT)

# load in the camera intrinsics
K = pickle.load(open('../camera2/cameraMatrix.pkl', 'rb'))

# compute the essential matrix
E_FR_FL = K.T @ F_FR_FL @ K 
E_FR_FL /= E_FR_FL[-1, -1]

# E_FL_BL = K.T @ F_FL_BL @ K 
# E_FL_BL /= E_FL_BL[-1, -1]

# compute the extrinsics for camera 1 (use global == camera 1)
identity_extrinsics = np.array([
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0]
])

P1 = K @ identity_extrinsics

# compute the possibilities for the extrinsic parameters for frontright and backleft cameras

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

extrinsic_options_FR = camera2(E_FR_FL)
# extrinsic_options_BL = camera2(E_FL_BL)

# use these possibilities to reconstruct the points 4 times
pts3d = np.zeros((4, NUM_PIXELS, 3))
final_pts3d = np.zeros((NUM_PIXELS, 3))
for i in range(extrinsic_options_FR.shape[2]):
  P2 = K @ extrinsic_options_FR[:, :, i]
  pts = cv.triangulatePoints(P1, P2, shared_pts1.T, shared_pts2.T).T

  # divide by final coordinate (homogeneous) and remove the final coord
  pts3d[i, FL_FR_shared] = (pts / np.expand_dims(pts[:, -1], -1))[:, :3]

# find the extrinsics that put the points in front of the camera
onlyz = pts3d[:, FL_FR_shared][:, :, 2]
positive_z_sums = np.sum(onlyz >= 0, axis=1)
best_idx = np.argmax(positive_z_sums)
final_pts3d[FL_FR_shared] = pts3d[best_idx][FL_FR_shared]

# only find the points that we actually need to triangulate

# needed = (filled1 & filled3) & np.logical_not(filled1 & filled2)
# # needed = (filled1 & filled3)
# needed_pts1 = pts1[needed]
# needed_pts3 = pts3[needed]

# do the same thing for the other pair
# pts3d = np.zeros((4, NUM_PIXELS, 3))
# for i in range(extrinsic_options_BL.shape[2]):
#   P2 = K @ extrinsic_options_BL[:, :, i]
#   pts = cv.triangulatePoints(P1, P2, needed_pts1.T, needed_pts3.T).T

#   # divide by final coordinate (homogeneous) and remove the final coord
#   pts3d[i, needed] = (pts / np.expand_dims(pts[:, -1], -1))[:, :3]

# # find the extrinsics that put the points in front of the camera
# onlyz = pts3d[:, needed][:, :, 2]
# positive_z_sums = np.sum(onlyz >= 0, axis=1)
# best_idx = np.argmax(positive_z_sums)
# final_pts3d[needed] = pts3d[best_idx][needed]

# # truncate to only the points we know
# final_pts3d = final_pts3d[FL_FR_shared | FL_BR_shared]

# find the reprojection error (for testing purposes)
# pts3_homo = np.append(final_pts3d, np.ones((final_pts3d.shape[0], 1)), axis=1)
# reproj_pts1 = np.squeeze(P1 @ np.expand_dims(pts3_homo, -1))
# reproj_pts1 = (reproj_pts1 / reproj_pts1[:, 2, np.newaxis])[:, :-1]
# # reproj_error = np.mean(np.linalg.norm(pts1[FL_FR_shared | FL_BR_shared] - reproj_pts1, axis=1))
# reproj_error = np.mean(np.linalg.norm(pts1 - reproj_pts1, axis=1))
# print(f'Reprojection error: {reproj_error}')

# swap them so z is the last column
final_pts3d[:, [2, 1]] = final_pts3d[:, [1, 2]]

# negate the z axis
final_pts3d[:, 2] *= -1

# center the points around the mean and make max distance be sqrt 2
mean_pt = np.mean(final_pts3d, axis=0)
final_pts3d -= mean_pt # center around the mean
max_dist = np.max(np.linalg.norm(final_pts3d, axis=1))
final_pts3d = (final_pts3d / max_dist) * np.sqrt(2)

# save the final points
np.save('../points/3dpoints_1_guess.npy', final_pts3d)


# display the points
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.scatter(final_pts3d[:, 0], final_pts3d[:, 1], final_pts3d[:, 2])
ax.set_aspect('equal')
plt.show()


'''
Next steps:

0. check if our labeled points are still valid with new undistortion params
1. label more points... should I use the old undistortion parameters? I guess
I can for now, but should prob verify that they aren't too different


The points do come out in the reference frame of the first camera, which is why we
had to check for the points being in front of that camera

'''