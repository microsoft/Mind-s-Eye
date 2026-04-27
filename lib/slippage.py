import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random, os
import random
import string
import json
from PIL import Image, ImageOps
from io import BytesIO
import matplotlib.image as mpimg


def random_shape(ax, x, y, size, shape_type=None, fill=True, lw=1):
    shape_type = shape_type or random.choice([
        'circle', 'square', 'triangle', 'pentagon', 'hexagon',
        'star', 'diamond', 'ellipse'
    ])
    
    kwargs = {'fill': fill, 'edgecolor': 'black', 'linewidth': lw}
    if fill:
        kwargs['facecolor'] = 'black'
    
    if shape_type == 'circle':
        ax.add_patch(patches.Circle((x, y), size, **kwargs))
    
    elif shape_type == 'square':
        ax.add_patch(patches.Rectangle((x - size, y - size), 2*size, 2*size, **kwargs))
    
    elif shape_type == 'triangle':
        ax.add_patch(patches.RegularPolygon((x, y), numVertices=3, radius=size, orientation=np.pi/2, **kwargs))
    
    elif shape_type == 'pentagon':
        ax.add_patch(patches.RegularPolygon((x, y), numVertices=5, radius=size, orientation=np.pi/2, **kwargs))
    
    elif shape_type == 'hexagon':
        ax.add_patch(patches.RegularPolygon((x, y), numVertices=6, radius=size, orientation=0, **kwargs))
    
    elif shape_type == 'diamond':
        diamond = np.array([
            (x, y + size), 
            (x + size, y), 
            (x, y - size), 
            (x - size, y)
        ])
        ax.add_patch(patches.Polygon(diamond, closed=True, **kwargs))
    
    elif shape_type == 'star':
        # 5-pointed star
        num_points = 5
        theta = np.linspace(0, 2 * np.pi, num_points * 2, endpoint=False)
        r = np.array([size, size / 2] * num_points)
        points = np.stack((r * np.cos(theta) + x, r * np.sin(theta) + y), axis=1)
        ax.add_patch(patches.Polygon(points, closed=True, **kwargs))
    
    elif shape_type == 'ellipse':
        ax.add_patch(patches.Ellipse((x, y), width=2*size, height=size, **kwargs))

def draw_spacing(ax, conform=True, variation=0):
    count = random.randint(3, 6)
    spacing = random.uniform(0.1, 0.3)
    y = 0.5
    shape = [
        'circle', 'square', 'triangle', 'pentagon', 'hexagon',
        'star', 'diamond', 'ellipse'
    ][variation % 8]
    if conform:
        start_x = 0.5 - (count - 1) * spacing / 2
        for i in range(count):
            x = start_x + i * spacing
            random_shape(ax, x, y, 0.04, shape_type=shape)
    else:
        for _ in range(count):
            x = np.random.uniform(0.1, 0.9)
            random_shape(ax, x, y, 0.04)

def draw_alignment(ax, conform=True, variation=0):
    count = random.randint(3, 6)
    shape = [
        'circle', 'square', 'triangle', 'pentagon', 'hexagon',
        'star', 'diamond', 'ellipse'
    ][variation % 8]
    if conform:
        y = 0.5
        for i in range(count):
            x = 0.2 + i * 0.12
            random_shape(ax, x, y, 0.04, shape_type=shape)
    else:
        for _ in range(count):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            random_shape(ax, x, y, 0.04)

def draw_number(ax, conform=True, variation=0):
    count = 3 if conform else random.choice([5, 6])
    shape = [
        'circle', 'square', 'triangle', 'pentagon', 'hexagon',
        'star', 'diamond', 'ellipse'
    ][variation % 8]
    for _ in range(count):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        random_shape(ax, x, y, 0.04, shape_type=shape)

def draw_enclosure(ax, conform=True, variation=0):
    outer_shape = ['circle', 'square','triangle','hexagon'][variation % 4]
    inner_shape = [
        'circle', 'square', 'triangle', 'pentagon', 'hexagon',
        'star', 'diamond', 'ellipse'
    ][variation % 8]
    x, y = 0.5, 0.5
    if conform:
        if outer_shape == 'circle':
            ax.add_patch(patches.Circle((x, y), 0.12, fill=False, edgecolor='black'))
        else:
            ax.add_patch(patches.Rectangle((x - 0.12, y - 0.12), 0.24, 0.24, fill=False, edgecolor='black'))
        random_shape(ax, x, y, 0.05, shape_type=inner_shape)
    else:
        random_shape(ax, 0.3, 0.3, 0.05, shape_type=inner_shape)
        ax.add_patch(patches.Circle((0.7, 0.7), 0.12, fill=False, edgecolor='black'))

def draw_symmetry(ax, conform=True, variation=0):
    shape = [
        'circle', 'square', 'triangle', 'pentagon', 'hexagon',
        'star', 'diamond', 'ellipse'
    ][variation % 8]

    if conform:
        # Choose one symmetry type
        symmetry_type = random.choice(['vertical', 'horizontal', 'diagonal', 'rotational'])

        if symmetry_type == 'vertical':
            for _ in range(2):
                x = np.random.uniform(0.1, 0.4)
                y = np.random.uniform(0.2, 0.8)
                random_shape(ax, x, y, 0.04, shape_type=shape)
                random_shape(ax, 1 - x, y, 0.04, shape_type=shape)

        elif symmetry_type == 'horizontal':
            for _ in range(2):
                x = np.random.uniform(0.2, 0.8)
                y = np.random.uniform(0.1, 0.4)
                random_shape(ax, x, y, 0.04, shape_type=shape)
                random_shape(ax, x, 1 - y, 0.04, shape_type=shape)

        elif symmetry_type == 'diagonal':
            for _ in range(2):
                x = np.random.uniform(0.1, 0.4)
                y = np.random.uniform(0.1, 0.4)
                random_shape(ax, x, y, 0.04, shape_type=shape)
                random_shape(ax, y, x, 0.04, shape_type=shape)  # mirror across y = x

        elif symmetry_type == 'rotational':
            for _ in range(2):
                x = np.random.uniform(0.2, 0.4)
                y = np.random.uniform(0.2, 0.4)
                random_shape(ax, x, y, 0.04, shape_type=shape)
                random_shape(ax, 1 - x, 1 - y, 0.04, shape_type=shape)  # 180° rotation

    else:
        # Asymmetric: no mirroring
        for _ in range(6):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            random_shape(ax, x, y, 0.04, shape_type=shape)


def draw_connectivity(ax, conform=True, variation=0):
    shape_count = 5 + variation % 3  # 3 to 7 shape pairs
    pattern = random.choice(['horizontal', 'z', 'inv_z', 'v', 'u'])

    x_start = 0.2
    x_step = 0.15
    x1s = [x_start + i * x_step for i in range(shape_count)]
    
    # Base y positions for different patterns
    if pattern == 'horizontal':
        y1s = [0.5] * shape_count
    elif pattern == 'z':
        y1s = [0.7 if i % 2 == 0 else 0.3 for i in range(shape_count)]
    elif pattern == 'inv_z':
        y1s = [0.3 if i % 2 == 0 else 0.7 for i in range(shape_count)]
    elif pattern == 'v':
        mid = shape_count // 2
        y1s = [0.7 - abs(i - mid) * 0.1 for i in range(shape_count)]
    elif pattern == 'u':
        mid = shape_count // 2
        y1s = [0.3 + abs(i - mid) * 0.1 for i in range(shape_count)]

    for i in range(shape_count):
        shape = [
            'circle', 'square', 'triangle', 'pentagon', 'hexagon',
            'star', 'diamond', 'ellipse'
        ][i % 8]

        x1 = x1s[i]
        y1 = y1s[i]
        x2 = x1 + 0.07
        y2 = y1

        # Draw first shape
        random_shape(ax, x1, y1, 0.03, shape_type=shape)

        if conform:
            # Second shape aligned with first
            random_shape(ax, x2, y2, 0.03, shape_type=shape)
            ax.plot([x1, x2], [y1, y2], color='black', lw=2)
        else:
            # Second shape offset, breaking connectivity
            offset_y = y2 + 0.15 if y2 < 0.5 else y2 - 0.15
            random_shape(ax, x2, offset_y, 0.03, shape_type=shape)
            # No connecting line


def draw_boolean_logic(ax, conform=True, variation=0):
    if conform:
        random_shape(ax, 0.4, 0.5, 0.05, shape_type='circle')
        random_shape(ax, 0.6, 0.5, 0.05, shape_type='triangle')
    else:
        shape = ['circle', 'triangle'][variation % 2]
        random_shape(ax, 0.5, 0.5, 0.07, shape_type=shape)


def draw_word_symmetry(ax, conform=True, word=None):
    if word is None:
        word = ''.join(random.choices(string.ascii_uppercase, k=random.randint(4, 6)))

    # Create an off-screen figure to render the text
    fig_temp, ax_temp = plt.subplots(figsize=(6, 2), dpi=150)
    ax_temp.set_xlim(0, 1)
    ax_temp.set_ylim(0, 1)
    ax_temp.axis('off')
    ax_temp.set_aspect('equal')

    # Draw the word with spacing
    letter_spacing = 0.2
    start_x = 0.1
    y_center = 0.5
    font_size = 20

    for i, letter in enumerate(word):
        x = start_x + i * letter_spacing + np.random.uniform(-0.005, 0.005)
        y = y_center + np.random.uniform(-0.01, 0.01)
        ax_temp.text(x, y, letter, fontsize=font_size,
                     va='center', ha='center', fontweight='bold')

    # Save to buffer
    buf = BytesIO()
    plt.tight_layout()
    fig_temp.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close(fig_temp)
    buf.seek(0)

    # Load image and mirror if needed
    img = Image.open(buf).convert("RGBA")
    if not conform:
        img = ImageOps.mirror(img)

    # Convert to array and display in the original axes
    img_array = np.array(img)

    # Insert the image back into the main plot (ax)
    ax.imshow(img_array, extent=(0, 1, 0, 1), aspect='auto')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_hollowness(ax, conform=True, variation=0):
    shape = [
        'circle', 'square', 'triangle', 'pentagon', 'hexagon',
        'star', 'diamond', 'ellipse'
    ][variation % 8]
    for i in range(3 + variation % 8):
        x = 0.2 + i * 0.2
        fill = not conform
        random_shape(ax, x, 0.5, 0.05, shape_type=shape, fill=not fill)

def draw_topology(ax, conform=True, variation=0):
    if conform:
        for _ in range(3 + variation % 2):
            x, y = np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)
            random_shape(ax, x, y, 0.04, shape_type='circle')
    else:
        for _ in range(3):
            x1, y1 = np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)
            x2, y2 = x1 + 0.1*np.random.randn(), y1 + 0.1*np.random.randn()
            ax.plot([x1, x2], [y1, y2], color='black', lw=2)

def draw_border(ax, conform=True, variation=0):
    lw = 4 if conform else 0.5
    shape = [
        'circle', 'square', 'triangle', 'pentagon', 'hexagon',
        'star', 'diamond', 'ellipse'
    ][variation % 8]
    for i in range(3 + variation % 2):
        x = 0.2 + i * 0.2
        random_shape(ax, x, 0.5, 0.05, shape_type=shape, fill=False, lw=lw)
