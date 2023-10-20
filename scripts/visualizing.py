import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Create a figure and a 3D axis
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Define the number of points and the number of frames for the animation
num_points = 500
num_frames = 100

# Generate random initial coordinates and colors for the points
x = np.random.rand(num_points)
y = np.random.rand(num_points)
z = np.random.rand(num_points)
colors = np.random.rand(num_points)

# Initialize the scatter plot with colors
scatter = ax.scatter(x, y, z, c=colors, cmap='viridis')

# Define an update function for the animation to change colors
def update(frame):
  colors = np.random.rand(num_points)  # Generate new random colors
  scatter.set_array(colors)  # Update the colors of the scatter plot

# Create the animation
ani = FuncAnimation(fig, update, frames=num_frames, repeat=False)

# Add a color bar to show the mapping of colors
# cbar = fig.colorbar(scatter, ax=ax, label='Color')

# Show the animation
plt.grid(False)
plt.axis('off')
# plt.rcParams['figure.figsize'] = [4, 4]
plt.show()
