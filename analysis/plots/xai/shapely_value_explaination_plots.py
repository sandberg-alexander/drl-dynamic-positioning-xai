import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np

def create_cut_corner_rectangle(center_x, center_y, width, height, corner_cut, angle, ax,
                               facecolor='light_blue', triangel_color='black', edgecolor=None, 
                               linestyle='-', alpha=1.0, add_thrusters=True, explaine=False):
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
    polygon = patches.Polygon(vertices, closed=True, 
                             facecolor=facecolor,
                             edgecolor=edgecolor if edgecolor else facecolor,
                             linestyle=linestyle,
                             linewidth=1.0*scale, alpha=alpha)
    
    # Create transformation: first rotate, then translate to center position
    t = transforms.Affine2D().rotate_deg(angle).translate(center_x, center_y) + ax.transData
    polygon.set_transform(t)
    
    # Add the polygon to the plot
    ax.add_patch(polygon)
    
    # Add a triangle at the "top" of the patch
    # For a -90 degree rotation, the "top" is towards the -y direction
    triangle_size = height * 0.1  # Size relative to height
    
    # Define triangle vertices relative to the rectangle
    triangle_vertices = np.array([
        [0, - half_h+0.01],           # Top point of triangle
        [-triangle_size/2, -half_h + triangle_size],  # Bottom left
        [triangle_size/2, -half_h + triangle_size]    # Bottom right
    ])
    
    # Create triangle polygon
    triangle = patches.Polygon(triangle_vertices, closed=True,
                              facecolor=triangel_color, edgecolor=triangel_color,
                              linewidth=1.0, alpha=1.0)
    
    # Apply the same transformation
    triangle.set_transform(t)
    
    # Add the triangle to the plot
    ax.add_patch(triangle)
    
    # Add thruster positions and vectors
    if add_thrusters:
        # Define thruster positions relative to the rectangle's center
        # Scale to match the rectangle dimensions
        Lx = half_w * 0.8  # Thrusters at 80% of half-width
        Ly = half_h * 0.8  # Thrusters at 80% of half-height
        
        thruster_positions = np.array([
            [Lx, Ly],       # Top-right
            [Lx, -Ly],      # Bottom-right
            [-Lx, Ly],      # Top-left
            [-Lx, -Ly]      # Bottom-left
        ])
        
        # Define thrust vector angles relative to rectangle orientation
        # Example: each thruster points outward at 45 degrees
        thruster_angles = np.array([
            270-0,    # bottom-left thruster
            270-(-(90-27)),   # top-left thruster
            270-0,   # bottom-right thruster
            270-(-(90-27))   # top-right thruster
        ])
        
        # Define thrust vector magnitudes (example values)
        thruster_magnitudes = np.array([.4, .4*np.sqrt(5.25), .4, .4*np.sqrt(5.25)])

        all_thrusters = [[thruster_angles, thruster_magnitudes, "tab:orange"]]

        if explaine:
            thruster_angles2 = np.array([
            270-(82.5-90),    # bottom-left thruster
            270-(-13.64-90),   # top-left thruster
            270-(82.5-90),   # bottom-right thruster
            270-(-13.64-90)   # top-right thruster
            ])
            
            # Define thrust vector magnitudes (example values)
            thruster_magnitudes2 = np.array([.4*.1, .4*.84, .4*.1, .4*.84])

            all_thrusters.append([thruster_angles2, thruster_magnitudes2, "tab:olive"])

        # Rotation in radians for thrust vectors
        rotation_rad = np.deg2rad(angle)
        
        # Rotation matrix for the rectangle's orientation
        rotation_matrix = np.array([
            [np.cos(rotation_rad), -np.sin(rotation_rad)],
            [np.sin(rotation_rad), np.cos(rotation_rad)]
        ])
        
        # Rotate thruster positions
        rotated_thrusters = rotation_matrix @ thruster_positions.T
        
        # Translate to center
        global_thruster_x = rotated_thrusters[0, :] + center_x
        global_thruster_y = rotated_thrusters[1, :] + center_y
        
        for thruster_angles, thruster_magnitudes, color in all_thrusters:

            # Plot thruster positions and vectors
            for i in range(4):
                # Plot thruster position
                ax.plot(global_thruster_x[i], global_thruster_y[i], 'ko', markersize=3)
                
                # Calculate thrust vector direction (in global coordinates)
                vector_angle = rotation_rad + np.deg2rad(thruster_angles[i])
                dx = np.cos(vector_angle) * thruster_magnitudes[i]
                dy = np.sin(vector_angle) * thruster_magnitudes[i]
                
                # Plot thrust vector
                color = color if facecolor != 'none' else "red"
                ax.quiver(global_thruster_x[i], global_thruster_y[i], dx, dy, 
                        angles='xy', scale_units='xy', scale=1, 
                        color=color, alpha=1.0, width=0.008*scale)
    
    return polygon

# Styling parameters
scale = 1/0.6
figsize = (4.8, 4.8)  # Adjusted height to match your example

# Create figure and axis
fig, ax = plt.subplots(figsize=figsize)

# Parameters for the rectangle
center_x, center_y = -1, -1  # Center position
width, height = 2.86/5.06, 1  # Size
corner_cut = 0.5/5.06  # Size of corner cut (adjusted)
angle = 180-90  # Rotation angle in degrees

# Create the rotated rectangle with cut corners (blue filled)
cut_rect = create_cut_corner_rectangle(
    center_x, center_y, width, height, corner_cut, angle, ax, 
    facecolor='tab:blue', edgecolor='none', alpha=1, explaine=True
)

# Create a second rectangle with red dashed outline and no fill
dashed_rect = create_cut_corner_rectangle(
    0, 0, width, height, corner_cut, 180, ax,
    facecolor='none', edgecolor='tab:red', triangel_color='tab:red', linestyle='--', alpha=1.0, add_thrusters=False
)

# Plot the center point
ax.plot(center_x, center_y, 'ko', markersize=3)
# Plot the center point
ax.plot(0, 0, 'o', markersize=3, color='tab:red')

# Set axis limits with padding
ax.set_xlim(-2, 1)
ax.set_ylim(-2, 1)  # Adjusted to match your example

# Add grid and labels with consistent sizing
ax.set_xlabel(r'$y$', fontsize=10*scale)
ax.set_ylabel(r'$x$', fontsize=10*scale)
ax.set_title(r'$x=-1, y=-1, \psi=90$', fontsize=10*scale)

# Set tick label size
ax.tick_params(axis='x', labelsize=8*scale)
ax.tick_params(axis='y', labelsize=8*scale)

# Set ticks at intervals of 1
ax.set_xticks(np.arange(-1, 0.1, 1))  # Ticks at -2, -1, 0
ax.set_yticks(np.arange(-1, 0.1, 1))  # Ticks at -2, -1, 0

# Add legend
handles = [
    plt.Line2D([0], [0], color='blue', lw=4, alpha=0.8, label='Current Position'),
    plt.Line2D([0], [0], color='red', linestyle='--', lw=2, label='Reference Position'),
    plt.Line2D([0], [0], color='lime', lw=2, label='Thrust Vector')
]
#ax.legend(handles=handles, loc='upper right', fontsize=10*scale)

# Use tight layout for better spacing
plt.tight_layout()

# Save the figure as PDF with high resolution
plt.savefig('shapley_value_explaination_plot_explain_psi.pdf', format='pdf', dpi=1200)

# Display the plot
plt.show()