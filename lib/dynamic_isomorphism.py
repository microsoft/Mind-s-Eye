import numpy as np
import matplotlib.pyplot as plt
import os
import random
# Output directory
num_frames = 6
t_list = np.linspace(0, 2 * np.pi, num_frames)

# Geometric transformation functions
def rotate(shape, angle):
    theta = np.radians(angle)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return shape @ rot_matrix.T

def translate(shape, dx, dy):
    return shape + np.array([dx, dy])

def scale(shape, factor):
    return shape * factor

def shear(shape, factor):
    shear_matrix = np.array([[1, factor], [0, 1]])
    return shape @ shear_matrix.T

def mirror_x(shape,x):
    return shape * np.array([1, -1])

def mirror_y(shape,x):
    return shape * np.array([-1, 1])

def flip(shape,x):
    return rotate(shape, 180)

def stretch_x(shape, factor):
    return shape * np.array([factor, 1])

def stretch_y(shape, factor):
    return shape * np.array([1, factor])

def contract_x(shape, factor):
    return shape * np.array([factor, 1])

def contract_y(shape, factor):
    return shape * np.array([1, factor])

def rotate_back_and_forth(shape, step):
    return rotate(shape, np.sin(step) * 30)

def bounce(shape, step):
    return translate(shape, 0, np.sin(step) * 0.5)

def wiggle(shape, step):
    return translate(shape, np.sin(step) * 0.5, 0)

def swirl(shape, step):
    angle = np.degrees(step)
    factor = 1 + 0.1 * np.sin(step)
    return scale(rotate(shape, angle), factor)

def pulsate(shape, step):
    factor = 1 + 0.2 * np.sin(step)
    return scale(shape, factor)

def spin_and_bounce(shape, step):
    return bounce(rotate(shape, np.degrees(step)), step)

def tilt(shape, step):
    return shear(shape, 0.5 * np.sin(step))

def compress_and_stretch(shape, step):
    return stretch_x(shape, 1 + 0.5 * np.sin(step))

def jump(shape, step):
    y_offset = 0.5 if (int(step * 3) % 2 == 0) else 0
    return translate(shape, 0, y_offset)

def vertical_oscillate(shape, t, amplitude=1.0, freq=1.0):
    offset = amplitude * np.sin(2 * np.pi * freq * t)
    return shape + np.array([0, offset])
# Transformation pair list
transform_pairs = [
    ("rotate_back_and_forth", "bounce"),
    ("wiggle", "bounce"),
    ("pulsate", "swirl"),
    ("spin_and_bounce", "bounce"),
    ("tilt", "rotate_back_and_forth"),
    ("mirror_x", "mirror_y"),
    ("flip", "flip"),
    ("stretch_x", "contract_x"),
    ("stretch_y", "contract_y"),
    ("compress_and_stretch", "bounce"),
    ("bounce", "jump"),
    ("swirl","rotate" ),
    ("scale", "contract_x"),
    ("contract_y", "stretch_y"),
    ("wiggle", "pulsate"),
    ("tilt", "swirl"),
    ("jump", "rotate"),
    ("mirror_x", "tilt"),
    ("flip", "rotate_back_and_forth"),
    ("scale", "bounce"),
]
# Map of transformation names to functions
# transform_map = {
#     "rotate_back_and_forth": lambda shape, t: rotate_back_and_forth(shape, t * 30),
#     "bounce": lambda shape, t: bounce(shape, t),
#     "wiggle": lambda shape, t: wiggle(shape,t),
#     "pulsate": lambda shape, t: pulsate(shape,t),
#     "swirl": lambda shape, t: swirl(shape, t),
#     "spin_and_bounce": lambda shape, t: spin_and_bounce(shape, t),
#     "tilt": lambda shape, t: tilt(shape, t),
#     "mirror_x": lambda shape, t: mirror_x(shape, t),
#     "mirror_y": lambda shape, t: mirror_y(shape, t),
#     "stretch_x": lambda shape, t: stretch_x(shape, t),
#     "contract_x": lambda shape, t: contract_x(shape, t),
#     "stretch_y": lambda shape, t: stretch_y(shape, t),
#     "contract_y": lambda shape, t: contract_y(shape, t),
#     "compress_and_stretch": lambda shape, t: compress_and_stretch(shape, t),
#     "bounce": lambda shape, t: bounce(shape, t),
#     "jump": lambda shape, t: jump(shape, t),
#     "rotate": lambda shape, step: rotate(shape, np.degrees(step * 1.5)),
#     "scale": lambda shape, step: scale(shape, 1.2),
#     "flip": lambda shape, t: flip(shape, t)
# }

transform_map = {
    "rotate_back_and_forth": lambda shape, t: rotate_back_and_forth(shape, t * 60),  # was 30
    "bounce": lambda shape, t: bounce(shape, t * 2),  # higher frequency
    "wiggle": lambda shape, t: wiggle(shape, t * 2),  # same
    "pulsate": lambda shape, t: pulsate(shape, t * 2),  # faster scale pulsation
    "swirl": lambda shape, t: swirl(shape, t * 2),  # faster spin and scale
    "spin_and_bounce": lambda shape, t: spin_and_bounce(shape, t * 2),  # bold spin + bounce
    "tilt": lambda shape, t: tilt(shape, t * 3),  # more shear
    "mirror_x": lambda shape, t: mirror_x(shape, t),
    "mirror_y": lambda shape, t: mirror_y(shape, t),
    "stretch_x": lambda shape, t: stretch_x(shape, 1.5 + np.sin(t * 3)),  # wider stretch
    "contract_x": lambda shape, t: contract_x(shape, 0.5 + 0.5 * np.abs(np.sin(t * 3))),  # sharper contract
    "stretch_y": lambda shape, t: stretch_y(shape, 1.5 + np.sin(t * 3)),
    "contract_y": lambda shape, t: contract_y(shape, 0.5 + 0.5 * np.abs(np.sin(t * 3))),
    "compress_and_stretch": lambda shape, t: compress_and_stretch(shape, t * 3),
    "jump": lambda shape, t: jump(shape, t * 3),  # more jumping
    "rotate": lambda shape, step: rotate(shape, np.degrees(step * 4)),  # faster spin
    "scale": lambda shape, step: scale(shape, 1 + 0.5 * np.sin(step * 3)),  # stronger pulsing
    "flip": lambda shape, t: flip(shape, t)
}




import numpy as np

def generate_structure(shape='triangle', scale=1.0, center=(0, 0)):
    """
    Generate a 2D shape with specified vertices, centered at 'center' with given 'scale'.
    
    Parameters:
    - shape (str): Type of shape ('triangle', 'square', 'pentagon', 'hexagon', 'diamond')
    - scale (float): Scaling factor for the shape (default: 1.0)
    - center (tuple): (x, y) coordinates for the shape's center (default: (0, 0))
    
    Returns:
    - points (ndarray): Array of shape (n_points, 2) containing the vertices
    """
    # Define number of vertices for each shape
    shape_to_vertices = {
        'triangle': 3,
        'square': 4,
        'pentagon': 5,
        'hexagon': 6,
        'diamond': 4  # Diamond is a square rotated 45 degrees
    }
    
    if shape not in shape_to_vertices:
        raise ValueError(f"Unsupported shape: {shape}. Choose from {list(shape_to_vertices.keys())}")
    
    n_points = shape_to_vertices[shape]
    points = np.zeros((n_points, 2))
    
    # Calculate angles for regular polygon vertices
    for i in range(n_points):
        # For diamond, start at 45 degrees; for others, start at 0
        angle_offset = np.pi / 4 if shape == 'diamond' else 0
        angle = angle_offset + (2 * np.pi * i) / n_points
        # Place points on a unit circle, then scale and translate
        points[i] = np.array([np.cos(angle), np.sin(angle)]) * scale + np.array(center)
    
    return points

# Shapes
triangle = np.array([[0, 0], [1, 0], [0.5, 1]])
diamond = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
def plot_grid(shape1, shape2, t_vals, func1, func2, filename, draw_axis=True):
    """
    Plot a 2x4 grid where top row shows first 4 transformations with func1 and func2,
    bottom row shows 5th transformation with func1 and func2 (labeled 'a') and
    three transformations with func1 and invalid_func (labeled 'b', 'c', 'd').
    
    Parameters:
    - shape1, shape2: Arrays of shape vertices
    - t_vals: List of time values for transformations
    - func1, func2: Transformation functions
    - filename: Output file path
    - draw_axis: Boolean to draw axis lines
    """
    # Ensure t_vals has at least 5 values
    if len(t_vals) < 5:
        raise ValueError("t_vals must contain at least 5 time values")
    
    # Create a 2x4 grid
    subplot_w, subplot_h = (12,12)
    figsize = (subplot_w * 4, subplot_h * 2)
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    
    # Top row: First 4 transformations with func1 and func2
    for idx in range(4):
        ax = axes[0, idx]
        ax.set_aspect('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        
        if draw_axis:
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        
        t = t_vals[idx]
        s1 = func1(shape1.copy(), t)
        s2 = func2(shape2.copy(), t)
        
        s1 = np.vstack([s1, s1[0]])
        s2 = np.vstack([s2, s2[0]])
        
        ax.plot(s1[:, 0], s1[:, 1], 'b-', label='Shape A',linewidth=8)
        ax.plot(s2[:, 0], s2[:, 1], 'r--', label='Shape B',linewidth=8)
        ax.set_title(f"t={t:.2f}", fontsize=40)
        ax.axis('off')
    
    # Bottom row: 5th transformation with func1 and func2 (label 'a')
    bottom_row_label = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    fifth_image_label = random.choice(list(bottom_row_label.keys()))
    remaining_labels = [label for label in bottom_row_label.keys() if label != fifth_image_label]
    subset_bottom_row_label = {key: bottom_row_label[key] for key in remaining_labels}
    ax = axes[1, fifth_image_label]
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    if draw_axis:
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    
    t = t_vals[4]
    s1 = func1(shape1.copy(), t)
    s2 = func2(shape2.copy(), t)
    
    s1 = np.vstack([s1, s1[0]])
    s2 = np.vstack([s2, s2[0]])
    
    ax.plot(s1[:, 0], s1[:, 1], 'b-', label='Shape A',linewidth=8)
    ax.plot(s2[:, 0], s2[:, 1], 'r--', label='Shape B',linewidth=8)
    ax.set_title(f"{bottom_row_label[fifth_image_label]}",fontsize=40)
    ax.axis('off')
    
    # Store the label of the 5th image
    fifth_image_label = f"({bottom_row_label[fifth_image_label]})"
    
    # Bottom row: Three transformations with func1 and invalid_func (labels 'b', 'c', 'd')
    def invalid_func(shape, t): return shape + np.array([np.cos(t * 5), np.sin(t * 5)])
    for idx, label in subset_bottom_row_label.items():
        ax = axes[1, idx]
        ax.set_aspect('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        
        if draw_axis:
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        
        t = t_vals[idx + 4] if idx + 4 < len(t_vals) else t_vals[-1]  # Use next t values or last one
        s1 = func1(shape1.copy(), t)
        s2 = invalid_func(shape2.copy(), t)
        
        s1 = np.vstack([s1, s1[0]])
        s2 = np.vstack([s2, s2[0]])
        
        ax.plot(s1[:, 0], s1[:, 1], 'b-', label='Shape A',linewidth=8)
        ax.plot(s2[:, 0], s2[:, 1], 'r--', label='Shape B',linewidth=8)
        ax.set_title(f"({label})",fontsize=40)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename,dpi=300)
    plt.close()
    
    return fifth_image_label
