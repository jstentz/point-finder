import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Create a figure and a 3D axis
screencolor = 'black'
fig = plt.figure(figsize=(10, 10), facecolor=screencolor)
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor(screencolor)

# Define the number of points and the number of frames for the animation
num_points = 500
num_frames = 1000000

# Generate random initial coordinates and colors for the points
x = 2 * np.random.rand(num_points) - 1
y = 2 * np.random.rand(num_points) - 1
z = 2 * np.random.rand(num_points) - 1
colors = np.random.rand(num_points, 3)
sizes = 100000 * np.ones(num_points)

# Initialize the scatter plot with colors
scatter = ax.scatter(x, y, z, c=colors, s=sizes, marker='o', edgecolors=None)

# Define an update function for the animation to change colors
def update(frame):
  colors = np.random.rand(num_points, 3)  # Generate new random colors
  scatter.set_color(colors)

# Create the animation
ani = FuncAnimation(fig, update, frames=num_frames, repeat=False)

# Add a color bar to show the mapping of colors
# cbar = fig.colorbar(scatter, ax=ax, label='Color')

# Show the animation
plt.grid(False)
plt.axis('off')
plt.show()
