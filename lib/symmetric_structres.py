import matplotlib.pyplot as plt
import numpy as np
import random
import os

def generate_random_filename():
    """Generate a random 8-digit filename with .png extension."""
    return ''.join(str(random.randint(0, 9)) for _ in range(8)) + '.png'

def generate_connected_path(n_segments=10, segment_length=0.5):
    """Generate a random connected path of line segments."""
    lines = []
    x, y = 0, 0  # Start at origin
    angle = 0
    for _ in range(n_segments):
        x2 = x + segment_length * np.cos(angle)
        y2 = y + segment_length * np.sin(angle)
        lines.append(((x, y), (x2, y2)))
        x, y = x2, y2
        angle += random.uniform(-np.pi/4, np.pi/4)  # Random turn
    return lines

def apply_symmetry(lines, symmetry='asymmetric', k=None):
    """Apply symmetry to a set of lines."""
    new_lines = lines.copy()
    if symmetry == 'vertical':
        # Reflect over y-axis
        for (x1, y1), (x2, y2) in lines:
            new_lines.append(((-x1, y1), (-x2, y2)))
    elif symmetry == 'horizontal':
        # Reflect over x-axis
        for (x1, y1), (x2, y2) in lines:
            new_lines.append(((x1, -y1), (x2, -y2)))
    elif symmetry == 'rotational':
        # Rotational symmetry of order k
        if k is None:
            k = 2
        for theta in [2 * np.pi * i / k for i in range(1, k)]:
            c, s = np.cos(theta), np.sin(theta)
            for (x1, y1), (x2, y2) in lines:
                # Rotate point 1
                x1r = x1 * c - y1 * s
                y1r = x1 * s + y1 * c
                # Rotate point 2
                x2r = x2 * c - y2 * s
                y2r = x2 * s + y2 * c
                new_lines.append(((x1r, y1r), (x2r, y2r)))
    # For 'asymmetric', return original lines
    return new_lines

def draw_lines(lines, ax):
    """Render lines on a given matplotlib axis."""
    try:
        for (x1, y1), (x2, y2) in lines:
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1)
        ax.set_aspect('equal')
        ax.axis('off')
    except Exception as e:
        print(f"Error in draw_lines: {e}")
        raise

def generate_structured_drawing_grid(output_dir):
    """Generate a 1x4 grid with 3 symmetric and 1 asymmetric drawing, labeled a-d."""
    # output_dir = "symmetric_structures_dataset"
    os.makedirs(output_dir, exist_ok=True)

    # Choose two symmetries
    symmetries = random.sample(['vertical', 'horizontal', 'rotational'], 2)
    # Create four sets of lines: 3 symmetric, 1 asymmetric
    drawings = []
    for sym in symmetries + [symmetries[0]]:  # Use first symmetry twice
        lines = generate_connected_path(
            n_segments=10,
            segment_length=0.5
        )
        k = random.choice([2, 4]) if sym == 'rotational' else None
        final_lines = apply_symmetry(lines, symmetry=sym, k=k)
        drawings.append(final_lines)
    # Add asymmetric drawing
    asymmetric_lines = generate_connected_path(
        n_segments=10,
        segment_length=0.5
    )
    drawings.append(asymmetric_lines)

    # Randomly shuffle drawings and assign labels
    labels = ['a', 'b', 'c', 'd']
    random.shuffle(drawings)
    asymmetric_index = drawings.index(asymmetric_lines)
    asymmetric_label = labels[asymmetric_index]

    # Create 1x4 grid
    try:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for ax, lines, label in zip(axes, drawings, labels):
            draw_lines(lines, ax)
            ax.set_title(label, fontsize=12)
        plt.tight_layout()
        random_filename = generate_random_filename()
        filename = os.path.join(output_dir, random_filename)
        # filename = os.path.join(output_dir, generate_random_filename())
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        print(f"The asymmetric drawing is labeled: {asymmetric_label}")
    except Exception as e:
        print(f"Error in generate_structured_drawing_grid: {e}")
        raise
    return asymmetric_label,random_filename
