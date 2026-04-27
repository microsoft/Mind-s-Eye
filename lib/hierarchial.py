import matplotlib.pyplot as plt
import numpy as np
import os
import random
import json

# -------------------------------
# Hierarchical structure functions
# -------------------------------

def nested_circles(ax, seed):
    np.random.seed(seed)
    for i in range(1, 4):
        radius = i * 0.2
        ax.add_patch(plt.Circle((0.5, 0.5), radius, fill=False, linewidth=2))

def grid_of_squares(ax, seed):
    np.random.seed(seed)
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((0.2 * i + 0.3, 0.2 * j + 0.3), 0.1, 0.1, fill=False, linewidth=2))

def nested_triangles(ax, seed):
    np.random.seed(seed)
    size = 0.4
    for i in range(3):
        offset = i * 0.05
        triangle = np.array([
            [0.5, 0.5 + size - offset],
            [0.5 - size + offset, 0.5 - size + offset],
            [0.5 + size - offset, 0.5 - size + offset],
        ])
        ax.plot(*triangle.T, 'k-', linewidth=2)
        ax.plot([triangle[-1][0], triangle[0][0]], [triangle[-1][1], triangle[0][1]], 'k-', linewidth=2)

def concentric_hexagons(ax, seed):
    np.random.seed(seed)
    for i in range(1, 4):
        angle = np.linspace(0, 2 * np.pi, 7)
        r = i * 0.1 + 0.1
        x = 0.5 + r * np.cos(angle)
        y = 0.5 + r * np.sin(angle)
        ax.plot(x, y, 'k-', linewidth=2)

def branching_lines(ax, seed):
    np.random.seed(seed)
    ax.plot([0.5, 0.5], [0.2, 0.8], 'k-', linewidth=2)
    ax.plot([0.5, 0.3], [0.5, 0.7], 'k-', linewidth=2)
    ax.plot([0.5, 0.7], [0.5, 0.7], 'k-', linewidth=2)

def stacked_rectangles(ax, seed):
    np.random.seed(seed)
    for i in range(3):
        ax.add_patch(plt.Rectangle((0.3, 0.3 + i * 0.12), 0.4, 0.1, fill=False, linewidth=2))

def recursive_arcs(ax, seed):
    np.random.seed(seed)
    for i in range(3):
        angle = np.linspace(0, np.pi, 100)
        r = 0.1 + i * 0.1
        x = 0.5 + r * np.cos(angle)
        y = 0.5 + r * np.sin(angle)
        ax.plot(x, y, 'k-', linewidth=2)

def layered_crosses(ax, seed):
    np.random.seed(seed)
    for i in range(3):
        offset = i * 0.05
        ax.plot([0.5 - 0.1 + offset, 0.5 + 0.1 - offset], [0.5, 0.5], 'k-', linewidth=2)
        ax.plot([0.5, 0.5], [0.5 - 0.1 + offset, 0.5 + 0.1 - offset], 'k-', linewidth=2)

def inward_spirals(ax, seed):
    np.random.seed(seed)
    theta = np.linspace(0, 4 * np.pi, 300)
    r = np.linspace(0.4, 0.05, 300)
    x = 0.5 + r * np.cos(theta)
    y = 0.5 + r * np.sin(theta)
    ax.plot(x, y, 'k-', linewidth=2)

def tree_structure(ax, seed):
    np.random.seed(seed)
    def draw_branch(x, y, angle, depth, scale=0.15):
        if depth == 0:
            return
        dx = scale * np.cos(angle)
        dy = scale * np.sin(angle)
        x_end = x + dx
        y_end = y + dy
        if 0 <= x_end <= 1 and 0 <= y_end <= 1:
            ax.plot([x, x_end], [y, y_end], 'k-', linewidth=2)
        draw_branch(x_end, y_end, angle - np.pi/6, depth - 1, scale)
        draw_branch(x_end, y_end, angle + np.pi/6, depth - 1, scale)
    draw_branch(0.5, 0.2, np.pi/2, 3)

def plot_tree(ax, seed):
    np.random.seed(seed)
    def draw_tree(x, y, angle, depth, scale=0.1):
        if depth == 0:
            return
        length = scale * depth
        x_end = x + np.cos(angle) * length
        y_end = y + np.sin(angle) * length
        if 0 <= x_end <= 1 and 0 <= y_end <= 1:
            ax.plot([x, x_end], [y, y_end], color='black')
        draw_tree(x_end, y_end, angle - np.pi / 6, depth - 1, scale)
        draw_tree(x_end, y_end, angle + np.pi / 6, depth - 1, scale)
    draw_tree(0.5, 0.5, np.pi / 2, 5)

def plot_sierpinski(ax, seed):
    np.random.seed(seed)
    def sierpinski(vertices, level):
        if level == 0:
            triangle = plt.Polygon(vertices, edgecolor='black', fill=None)
            ax.add_patch(triangle)
        else:
            midpoints = [
                (vertices[0] + vertices[1]) / 2,
                (vertices[1] + vertices[2]) / 2,
                (vertices[2] + vertices[0]) / 2,
            ]
            sierpinski([vertices[0], midpoints[0], midpoints[2]], level - 1)
            sierpinski([vertices[1], midpoints[0], midpoints[1]], level - 1)
            sierpinski([vertices[2], midpoints[1], midpoints[2]], level - 1)
    vertices = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.2 + np.sqrt(3)/2 * 0.6]])
    sierpinski(vertices, 3)  # Reduced depth for visibility

def plot_nested_squares(ax, seed):
    np.random.seed(seed)
    center = [0.5, 0.5]
    for i in range(6):
        size = 0.9 * (0.8 ** i)
        lower_left = [center[0] - size / 2, center[1] - size / 2]
        square = plt.Rectangle(lower_left, size, size, fill=None, edgecolor='black')
        ax.add_patch(square)

def plot_lsystem(ax, seed):
    np.random.seed(seed)
    def draw_lsystem(x, y, angle, length, depth):
        if depth == 0:
            return
        x_end = x + np.cos(angle) * length
        y_end = y + np.sin(angle) * length
        if 0 <= x_end <= 1 and 0 <= y_end <= 1:
            ax.plot([x, x_end], [y, y_end], color='black')
        draw_lsystem(x_end, y_end, angle + np.pi / 6, length * 0.7, depth - 1)
        draw_lsystem(x_end, y_end, angle - np.pi / 6, length * 0.7, depth - 1)
    draw_lsystem(0.5, 0.5, np.pi / 2, 0.2, 4)

def plot_nested_circles(ax, seed):
    np.random.seed(seed)
    center = [0.5, 0.5]
    for i in range(6):
        radius = 0.4 * (0.8 ** i)
        circle = plt.Circle(center, radius, fill=None, edgecolor='black')
        ax.add_patch(circle)

def plot_radial_tree(ax, seed):
    np.random.seed(seed)
    def branch(x, y, angle, radius, depth):
        if depth == 0:
            return
        for i in range(3):
            theta = angle + i * (2 * np.pi / 3)
            x_end = x + np.cos(theta) * radius
            y_end = y + np.sin(theta) * radius
            if 0 <= x_end <= 1 and 0 <= y_end <= 1:
                ax.plot([x, x_end], [y, y_end], color='black')
            branch(x_end, y_end, theta, radius * 0.5, depth - 1)
    branch(0.5, 0.5, 0, 0.2, 3)

def plot_fractal_cross(ax, seed):
    np.random.seed(seed)
    def draw_cross(x, y, size, depth):
        if depth == 0:
            return
        ax.plot([x - size, x + size], [y, y], color='black')
        ax.plot([x, x], [y - size, y + size], color='black')
        for dx, dy in [(-size, 0), (size, 0), (0, -size), (0, size)]:
            if 0 <= x + dx <= 1 and 0 <= y + dy <= 1:
                draw_cross(x + dx, y + dy, size * 0.5, depth - 1)
    draw_cross(0.5, 0.5, 0.2, 3)

def plot_hexagonal_nest(ax, seed):
    np.random.seed(seed)
    def draw_hex(x, y, size):
        for i in range(6):
            angle1 = np.pi / 3 * i
            angle2 = np.pi / 3 * (i + 1)
            x1 = x + size * np.cos(angle1)
            y1 = y + size * np.sin(angle1)
            x2 = x + size * np.cos(angle2)
            y2 = y + size * np.sin(angle2)
            if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
                ax.plot([x1, x2], [y1, y2], color='black')
    for i in range(3):
        draw_hex(0.5, 0.5, 0.4 * (0.8 ** i))

def plot_pythagoras_tree(ax, seed):
    np.random.seed(seed)
    def draw(x, y, size, angle, depth):
        if depth == 0:
            return
        x1 = x + size * np.cos(angle)
        y1 = y + size * np.sin(angle)
        x2 = x + size * np.cos(angle + np.pi/2)
        y2 = y + size * np.sin(angle + np.pi/2)
        x3 = x1 + size * np.cos(angle + np.pi/2)
        y3 = y1 + size * np.sin(angle + np.pi/2)
        vertices = [[x, y], [x1, y1], [x3, y3], [x2, y2]]
        if all(0 <= v[0] <= 1 and 0 <= v[1] <= 1 for v in vertices):
            square = plt.Polygon(vertices, fill=None, edgecolor='black')
            ax.add_patch(square)
        draw(x3, y3, size * 0.7, angle + np.pi/6, depth - 1)
        draw(x2, y2, size * 0.7, angle - np.pi/6, depth - 1)
    draw(0.4, 0.2, 0.2, 0, 4)

# All hierarchical structure functions
hierarchical_funcs = [
    nested_circles, grid_of_squares, nested_triangles, concentric_hexagons,
    stacked_rectangles, recursive_arcs, layered_crosses,
    inward_spirals, tree_structure, plot_tree,
    plot_sierpinski, plot_nested_squares, plot_lsystem,
    plot_nested_circles, plot_radial_tree,
    plot_fractal_cross, plot_hexagonal_nest, plot_pythagoras_tree
]

# A random non-hierarchical function for violations
def violate_structure(ax, seed):
    np.random.seed(seed)
    for _ in range(5):
        x = np.random.uniform(0, 1, 2)
        y = np.random.uniform(0, 1, 2)
        ax.plot(x, y, 'k-', linewidth=1.5)

def generate_random_filename():
    """Generate a random 8-digit filename with .png extension."""
    return ''.join(str(random.randint(0, 9)) for _ in range(8)) + '.png'

# -------------------------------
# Generate composite images
# -------------------------------