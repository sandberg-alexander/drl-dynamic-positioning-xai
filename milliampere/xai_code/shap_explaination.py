import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np

def create_cut_corner_rectangle(center_x, center_y, width, height, corner_cut, angle, ax,
                               facecolor='blue', alpha=1.0):
    """
    Create a rectangle with cut corners at a specific position and rotation
    """
    # Calculate half dimensions
    half_w = width / 2
    half_h = height / 2
    cut = corner_cut
    
    # Define vertices in counter-clockwise order (corrected)
    vertices = np.array([
        [-half_w + cut, -half_h],      # Bottom left corner cut
        [half_w - cut, -half_h],       # Bottom right corner cut
        [half_w, -half_h + cut],       # Bottom right after cut
        [half_w, half_h - cut],        # Top right corner cut
        [half_w - cut, half_h],        # Top right after cut
        [-half_w + cut, half_h],       # Top left after cut
        [-half_w, half_h - cut],       # Top left corner cut
        [-half_w, -half_h + cut]       # Bottom left after cut
    ])
    
    # Create polygon at origin
    polygon = patches.Polygon(vertices, closed=True, facecolor=facecolor,
                             linewidth=1.5, alpha=alpha)
    
    # Create transformation: first rotate, then translate to center position
    t = transforms.Affine2D().rotate_deg(angle).translate(center_x, center_y) + ax.transData
    polygon.set_transform(t)
    
    # Add the polygon to the plot
    ax.add_patch(polygon)
    
    # Add a triangle at the "top" of the patch
    # For a -90 degree rotation, the "top" is at what was the right side
    triangle_size = height * 0.15  # Size relative to height
    
    # Define triangle vertices relative to the rectangle
    # At right side center before rotation (becomes top after -90 rotation)
    triangle_vertices = np.array([
        [half_w + triangle_size, 0],       # Right point of triangle (becomes top)
        [half_w, triangle_size/2],         # Upper vertex
        [half_w, -triangle_size/2]         # Lower vertex
    ])
    
    # Create triangle polygon
    triangle = patches.Polygon(triangle_vertices, closed=True, 
                               facecolor='black', edgecolor='black', 
                               linewidth=1.0, alpha=1.0)
    
    # Apply the same transformation
    triangle.set_transform(t)
    
    # Add the triangle to the plot
    ax.add_patch(triangle)
    
    return polygon

# Styling parameters
scale = 1
figsize = (4.8, 4.8)  # Adjusted height to match your example

# Create figure and axis
fig, ax = plt.subplots(figsize=figsize)

# Parameters for the rectangle
center_x, center_y = -1, -1  # Center position
width, height = 2.86/5.06, 1  # Size
corner_cut = 0.5/5.06  # Size of corner cut (adjusted)
angle = -90  # Rotation angle in degrees

# Create the rotated rectangle with cut corners
cut_rect = create_cut_corner_rectangle(
    center_x, center_y, width, height, corner_cut, angle, ax, alpha=0.8
)

# Plot the center point
ax.plot(center_x, center_y, 'ko', markersize=3)

# Set axis limits with padding
ax.set_xlim(-2, .5)
ax.set_ylim(-2, .5)  # Adjusted to match your example

# Add grid and labels with consistent sizing
ax.set_xlabel(r'$y^n$', fontsize=14*scale)
ax.set_ylabel(r'$x^n$', fontsize=14*scale)
ax.set_title(r'$\varnothing$, $(x=0, y=0, \psi=0)$', fontsize=10*scale)

# Set tick label size
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)

# Set ticks at intervals of 1
ax.set_xticks(np.arange(-2, 0.1, 1))  # Ticks at -2, -1, 0
ax.set_yticks(np.arange(-2, 0.1, 1))  # Ticks at -2, -1, 0

# Use tight layout for better spacing
plt.tight_layout()

# Save the figure as PDF with high resolution
plt.savefig('rotated_cut_corner_rectangle.pdf', format='pdf', dpi=1200)

# Display the plot
plt.show()