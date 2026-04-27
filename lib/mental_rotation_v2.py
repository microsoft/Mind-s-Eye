#!/usr/bin/env python3
import os
import random
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec

CUBE_COLORS = [
    'red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange',
    'purple', 'lime', 'pink', 'teal', 'violet', 'brown', 'gold',
    'coral', 'navy', 'olive', 'maroon', 'turquoise', 'salmon'
]

# Global dictionary of polycube shapes
SHAPES = {
    "Snake": [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1),
        (1, 2, 1), (2, 2, 1), (2, 2, 2), (2, 3, 2),
    ],
    "Zigzag": [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0),
        (2, 1, 1), (2, 2, 1), (3, 2, 1), (3, 2, 2),
    ],
    "SnakeComplex1": [
        (0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0),
        (2, 1, 1), (2, 2, 1), (1, 2, 1), (1, 3, 1), (1, 3, 2),
    ],
    "HookedCorner": [
        (0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0),
        (0, 2, 0), (0, 2, 1), (0, 2, 2),
    ],
    "TopPlate": [
        (0, 0, 0), (0, 1, 0), (0, 2, 0),
        (0, 2, 1), (1, 2, 1), (2, 2, 1),
    ],
    "CornerStaircase": [
        (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0),
        (1, 1, 0), (2, 1, 0), (3, 1, 0), (3, 2, 0), (3, 3, 0),
    ],
    "TripleArm": [
        (3, -1, 0), (3, -1, 1), (3, -1, 2), (0, 0, 0),
        (1, 0, 0), (2, 0, 0), (3, 0, 0), (0, 1, 0), (0, 2, 0),
    ],
}

# Shapes available for each difficulty
EASY_SHAPES = ["Snake", "HookedCorner", "TopPlate", "CornerStaircase", "TripleArm"]
COMPLEX_SHAPES = list(SHAPES.keys())

# Dynamic similar-object mapping
all_shape_keys = list(SHAPES.keys())
SIMILAR_MAPPING = {
    key: [s for s in all_shape_keys if s != key] for key in all_shape_keys
}


def set_axes_equal(ax, all_vertices):
    """Make the aspect ratio equal and remove visual distractions."""
    all_vertices = np.array(all_vertices)
    x_limits = [np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])]
    y_limits = [np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])]
    z_limits = [np.min(all_vertices[:, 2]), np.max(all_vertices[:, 2])]
    
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)
    
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    
    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)
    ax.set_box_aspect([1, 1, 1])
    
    # Set ticks and remove labels
    ax.set_xticks(np.linspace(x_limits[0], x_limits[1], 5))
    ax.set_yticks(np.linspace(y_limits[0], y_limits[1], 5))
    ax.set_zticks(np.linspace(z_limits[0], z_limits[1], 5))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    # Remove 3D visual elements for complex mode
    if hasattr(ax, '_remove_3d_elements'):
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.grid(False)
        ax._axis3don = False


def cube_vertices(origin, size=1.0):
    """Return the 8 corner vertices of a cube."""
    x, y, z = origin
    return np.array([
        [x, y, z], [x + size, y, z], [x + size, y + size, z], [x, y + size, z],
        [x, y, z + size], [x + size, y, z + size], 
        [x + size, y + size, z + size], [x, y + size, z + size],
    ])


def plot_cubes(ax, vertices, cube_colors, title="", hide_3d_elements=False):
    """Plot cubes using their vertices with different colors for each cube."""
    if hide_3d_elements:
        ax._remove_3d_elements = True
    
    n_cubes = len(vertices) // 8
    vertices_reshaped = vertices.reshape((n_cubes, 8, 3))
    
    for cube_idx, cube_verts in enumerate(vertices_reshaped):
        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4],
        ]
        for face in faces:
            polygon = Poly3DCollection(
                [cube_verts[face]], facecolors=cube_colors[cube_idx % len(cube_colors)],
                edgecolors="black", alpha=1.0,
            )
            ax.add_collection3d(polygon)
    
    ax.set_title(title, fontsize=12)
    set_axes_equal(ax, vertices)


def generate_shape_vertices(shape_name, cube_size=1.0):
    """Generate vertices for a given shape."""
    if shape_name not in SHAPES:
        raise ValueError(f"Unknown shape {shape_name}")
    
    cube_origins = SHAPES[shape_name]
    all_vertices = []
    for origin in cube_origins:
        corners = cube_vertices(origin, size=cube_size)
        all_vertices.append(corners)
    return np.vstack(all_vertices)


def get_transformed_candidate(transformation_func, original, max_attempts=10):
    """Apply transformation until result differs from original."""
    for _ in range(max_attempts):
        candidate = transformation_func(original)
        if not np.allclose(candidate, original, atol=1e-6):
            return candidate
    return candidate

def get_transformed_candidate_v2(transformation_func, original, max_attempts=10):
    """Apply transformation until result differs from original."""
    for _ in range(max_attempts):
        candidate, angles = transformation_func(original)
        if not np.allclose(candidate, original, atol=1e-6):
            return candidate, angles
    return candidate, angles

def transform_rotate(vertices, difficulty="easy"):
    """Rotate shape based on difficulty level."""
    center = vertices.mean(axis=0)
    shifted = vertices - center
    
    if difficulty == "easy":
        # Single axis rotation with simple angles
        axis = random.choice(["x", "y", "z"])
        angle = np.deg2rad(random.choice([-90, 90, 180]))
        
        if axis == "x":
            R = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ])
        elif axis == "y":
            R = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ])
        else:  # z axis
            R = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ])
    else:  # complex
        # Multi-axis rotation with varied angles
        angle_x = np.deg2rad(np.random.choice([0, 60, 90, 120]))
        angle_y = np.deg2rad(np.random.choice([0, 60, 90, 120]))
        angle_z = np.deg2rad(np.random.choice([0, 60, 90, 120]))
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ])
        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)],
        ])
        Rz = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1],
        ])
        R = Rz @ Ry @ Rx
    
    rotated = (R @ shifted.T).T
    return rotated + center


def transform_rotate_v2(vertices, difficulty="easy"):
    """Rotate shape based on difficulty level and return angles."""
    center = vertices.mean(axis=0)
    shifted = vertices - center
    
    if difficulty == "easy":
        # Single axis rotation with simple angles
        axis = random.choice(["x", "y", "z"])
        angle = np.deg2rad(random.choice([-90, 90, 180]))
        
        angles = [0, 0, 0]
        if axis == "x":
            angles[0] = angle
            R = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ])
        elif axis == "y":
            angles[1] = angle
            R = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ])
        else:  # z axis
            angles[2] = angle
            R = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ])
    else:  # complex
        # Multi-axis rotation with varied angles
        angle_x = np.deg2rad(np.random.choice([0, 60, 90, 120]))
        angle_y = np.deg2rad(np.random.choice([0, 60, 90, 120]))
        angle_z = np.deg2rad(np.random.choice([0, 60, 90, 120]))
        
        angles = [angle_x, angle_y, angle_z]
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ])
        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)],
        ])
        Rz = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1],
        ])
        R = Rz @ Ry @ Rx
    
    rotated = (R @ shifted.T).T
    # print(angles)
    angles_deg = [str(np.round(np.rad2deg(angle))) for angle in angles]
    print(angles_deg) 
    return rotated + center, angles_deg


def transform_mirror(vertices, difficulty="easy"):
    """Mirror shape based on difficulty level."""
    center = vertices.mean(axis=0)
    shifted = vertices - center
    mirrored = shifted.copy()
    
    if difficulty == "easy":
        # Mirror across XY plane (Z-axis)
        mirrored[:, 2] = -mirrored[:, 2]
    else:  # complex
        # Mirror across random axis
        axis = random.choice([0, 1, 2])
        mirrored[:, axis] = -mirrored[:, axis]
    
    return mirrored + center


def get_visually_similar_candidate(chosen_shape_name, original_vertices, cube_size=1.0, difficulty="easy"):
    """Get a similar shape candidate."""
    if chosen_shape_name in SIMILAR_MAPPING:
        similar_candidates = SIMILAR_MAPPING[chosen_shape_name][:]
        random.shuffle(similar_candidates)
        
        for similar_shape_name in similar_candidates:
            if similar_shape_name in SHAPES:
                candidate_vertices = generate_shape_vertices(similar_shape_name, cube_size)
                candidate_vertices = get_transformed_candidate(
                    lambda v: transform_rotate(v, difficulty), candidate_vertices
                )
                if (candidate_vertices.shape != original_vertices.shape or 
                    not np.allclose(candidate_vertices, original_vertices, atol=1e-6)):
                    return candidate_vertices
    return None

import numpy as np
import random

CUBE_COLORS = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange']

def randomize_cube_colors(similar_candidate):
    """
    Randomly change the colors of cubes in a similar_candidate shape and return vertices with colors.
    
    Args:
        similar_candidate (np.ndarray): Array of vertices for the shape (n_cubes * 8, 3)
        cube_size (float): Size of each cube
    
    Returns:
        tuple: (vertices, cube_colors) where vertices is the input vertices and 
               cube_colors is a list of randomly assigned colors for each cube
    """
    # Validate input
    if not isinstance(similar_candidate, np.ndarray) or similar_candidate.shape[1] != 3:
        raise ValueError("similar_candidate must be a numpy array with shape (n, 3)")
    
    # Calculate number of cubes (each cube has 8 vertices)
    n_cubes = len(similar_candidate) // 8
    if len(similar_candidate) % 8 != 0:
        raise ValueError("Number of vertices must be a multiple of 8 (8 vertices per cube)")
    
    # Generate random colors for each cube
    cube_colors = [random.choice(CUBE_COLORS) for _ in range(n_cubes)]
    
    # Assign colors to each vertex by repeating the cube color for all 8 vertices of each cube
    vertex_colors = []
    for i in range(n_cubes):
        vertex_colors.extend([cube_colors[i]] * 8)
    
    # Combine vertices with their colors
    vertices_with_colors = [(similar_candidate[i], vertex_colors[i]) for i in range(len(similar_candidate))]
    
    return vertices_with_colors


def generate_one_image(index, difficulty="easy", outdir="data/mrt"):
    """Generate a single MRT image based on difficulty."""
    cube_size = 1.0
    
    # Select shapes based on difficulty
    shapes_list = EASY_SHAPES if difficulty == "easy" else COMPLEX_SHAPES
    shape_name = random.choice(shapes_list)
    original_vertices = generate_shape_vertices(shape_name, cube_size=cube_size)
    
    # Generate random colors for cubes
    n_cubes = len(SHAPES[shape_name])
    if difficulty == "easy" and index < 300:
        cube_colors = ["white" for _ in range(n_cubes)]
    else:
        cube_colors = [random.choice(CUBE_COLORS) for _ in range(n_cubes)]
    
    # Generate correct candidate (rotation)
    correct_candidate, angles = get_transformed_candidate_v2(
        lambda v: transform_rotate_v2(v, difficulty), original_vertices
    )
    
    # Generate wrong candidates
    mirror_candidate = get_transformed_candidate(
        lambda v: transform_rotate(transform_mirror(v, difficulty), difficulty),
        original_vertices
    )
    
    similar_candidate = get_visually_similar_candidate(
        shape_name, original_vertices, cube_size, difficulty
    )
    if similar_candidate is None:
        similar_candidate = mirror_candidate
    
    # Set up candidates based on difficulty
    if difficulty == "easy":
        mirror_candidate2 = get_transformed_candidate(
            lambda v: transform_rotate(transform_mirror(v, difficulty), difficulty),
            original_vertices
        )
        candidates = [
            ("rotate", correct_candidate),
            ("mirror", mirror_candidate),
            ("mirror2", mirror_candidate2),
            ("similar", similar_candidate),
        ]
        num_candidates = 4
        figure_size = (12, 8)
    else:  # complex
        mirror_candidate2 = get_transformed_candidate(
            lambda v: transform_rotate(transform_mirror(v, difficulty), difficulty),
            original_vertices
        )
        candidates = [
            ("rotate", correct_candidate),
            ("mirror", mirror_candidate),
            ("similar", similar_candidate),
            ("rotate2", correct_candidate),
        ]
        num_candidates = 4
        figure_size = (12, 8)
    
    random.shuffle(candidates)
    correct_candidate_index = [
        i for i, cand in enumerate(candidates) if cand[0] == "rotate"
    ][0]
    if difficulty == "complex":
        correct_candidate_index_for_color_perturb = [
            i for i, cand in enumerate(candidates) if cand[0] == "rotate2"][0]
    else:
        correct_candidate_index_for_color_perturb = -1

    # Create figure
    fig = plt.figure(figsize=figure_size)
    gs = GridSpec(2, num_candidates, height_ratios=[0.5, 1], wspace=0.1, hspace=0.1)
    
    # Plot original shape
    ax_orig = fig.add_subplot(gs[0, :], projection="3d")
    plot_cubes(ax_orig, original_vertices, cube_colors, title="Original Shape", 
               hide_3d_elements=(difficulty == "complex"))
    
    # Plot candidates
    shuffling_cube_colors = cube_colors
    for i in range(num_candidates):
        ax = fig.add_subplot(gs[1, i], projection="3d")
        _, candidate_vertices = candidates[i]
        if difficulty == "complex" and i == correct_candidate_index_for_color_perturb:
            shuffling_cube_colors = random.sample(cube_colors, len(cube_colors))
        else:
            shuffling_cube_colors = cube_colors
        plot_cubes(ax, candidate_vertices, shuffling_cube_colors, title=f"Option {chr(65 + i)}", 
                   hide_3d_elements=(difficulty == "complex"))
    
    # Save image
    filename = f"{shape_name}_{index}.png"
    output_path = os.path.join(outdir, filename)
    plt.savefig(output_path, dpi=60, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
    # Save metadata
    metadata = {
        "filename": filename,
        "difficulty": difficulty,
        "shape": shape_name,
        "correct_angles":angles,
        "candidate_order": [tag for tag, _ in candidates],
        "answer": chr(65 + correct_candidate_index),
    }
    return metadata

def main():
    parser = argparse.ArgumentParser(
        description="Generate mental rotation test images with variable difficulty."
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=69,
        help="Seed for reproducible results"
    )
    parser.add_argument(
        "--outdir", "-o", type=str, default=None,
        help="Output directory (defaults to data/mrt/{difficulty})"
    )
    
    args = parser.parse_args()
    
    # Set default output directory based on difficulty
    if args.outdir is None:
        args.outdir = f"data/mrt/{args.difficulty}"
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Set seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Generate images
    annotations = {}
    for i in range(100):
        if i < 50:
            single_annotation = generate_one_image(i, difficulty="easy", outdir=args.outdir)
        else:
            single_annotation = generate_one_image(i, difficulty="complex", outdir=args.outdir)

        annotations[single_annotation["filename"]] = single_annotation

    with open(os.path.join(args.outdir, "annotation.json"), "a") as f:
        f.write(json.dumps(annotations) + "\n")

    print(f"Generated MRT images in {args.outdir}")


# if __name__ == "__main__":
#     main()