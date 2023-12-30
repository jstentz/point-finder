import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# plt.ion()

# Create a figure and a 3D axis
screencolor = 'black'
fig = plt.figure(figsize=(10, 10), facecolor=screencolor)
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor(screencolor)

# Define the number of points and the number of frames for the animation
num_points = 500
num_frames = 1000000

# uniformly within a sphere
# points = np.zeros((num_points, 3))
# for i in range(num_points):
#   while np.linalg.norm(point := 2 * np.random.rand(3) - 1) > 1:
#     pass
#   points[i] = point

# load points
points = np.load('../points/3dpoints_1_guess.npy')

colors = np.zeros((num_points, 3))
sizes = 100 * np.ones(num_points)

# Initialize the scatter plot with colors
scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=sizes, marker='o', edgecolors=None, alpha=0.4)


# Define an update function for the animation to change colors
# def update(frame):
#   global colors
#   step = np.pi / 200
#   t = step * frame

#   colorin = (np.sin(3 * (points[:, 0] + t)) / 1.3) < points[:, 2]

#   colors[colorin] = np.array([0, 0, 0])
#   colors[np.logical_not(colorin)] = np.array([1, 1, 1])

#   blanks = np.logical_not(colors == np.array([0, 0, 0]))
#   scatter.set_color(colors)
#   # scatter.set_offsets(points * blanks)
#   scatter.set_offsets(np.zeros((num_points, 3)))
#   return scatter


# t = 0
# step = np.pi / 200
# epsilon = 0.1

# # Define an update function for the animation to change colors
# def update(frame):
#   global t, colors
#   t += step

#   colors[:, 0] = (np.sin(points[:, 0] + t) / 2) + 0.5
#   colors[:, 1] = (np.sin(points[:, 1] + t + np.pi) / 2) + 0.5
#   colors[:, 2] = (np.sin(points[:, 2] + t) / 2) + 0.5
#   scatter.set_color(colors)


angle = 0
step = np.pi / 200
epsilon = 0.1

# Define an update function for the animation to change colors
def update(frame):
  global angle, colors
  angle += step

  colors = np.sin(points + angle)**2
  scatter.set_color(colors)

# angle = 0
# step = np.pi / 200
# epsilon = 0.1

# # Define an update function for the animation to change colors
# def update(frame):
#   global angle, colors
#   colors *= 0.95
#   angle += step
#   plane1 = np.array([np.sin(angle), np.cos(angle), 0])
#   plane2 = np.array([np.sin(angle - np.pi/2), np.cos(angle - np.pi/2), 0])

#   # plane = np.array([0, 1, 0])
#   # color1 = np.array([1, 0, 0])
#   # color2 = np.array([0, 1, 0])
#   green = np.array([1, 0, 0])
#   red = np.array([0, 1, 0])
#   # colors = np.random.rand(num_points, 3)  # Generate new random colors
#   # colors = np.zeros((num_points, 3))
#   for i in range(colors.shape[0]):
#     if abs(np.dot(plane1, points[i])) < epsilon:
#       colors[i] = green
#     if abs(np.dot(plane2, points[i])) < epsilon:
#       colors[i] = red

#   blanks = np.logical_not(colors == np.array([0, 0, 0]))
#   scatter.set_color(colors)
#   scatter.set_offsets(points * blanks)

# # Create the animation
ani = FuncAnimation(fig, update, frames=num_frames, interval=10, repeat=False)

# # Show the animation
plt.grid(False)
plt.axis('off')
plt.show()

# import time
# frame = 0
# # this doesn't update anything bruh
# while True:
#   update(frame)
#   plt.draw()
#   frame += 1
#   time.sleep(0.1)