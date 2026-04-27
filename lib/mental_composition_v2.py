import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import string
import os
import json
# Mapping of 3D nets to their respective 3D shapes
hard_net_to_shape = {
    "cube_net": "cube",
    "tetrahedron_net": "tetrahedron",
    "octahedron_net": "octahedron",
    "dodecahedron_net": "dodecahedron",
    "icosahedron_net": "icosahedron",
    "cuboid_net": "cuboid",
    "triangular_prism_net": "triangular_prism",
    "square_pyramid_net": "square_pyramid",
    "cone_net": "cone"

}
easy_net_to_shape = {
    "cube_net": "cube",
    "cuboid_net": "cuboid",
    "triangular_prism_net": "triangular_prism",
    "square_pyramid_net": "square_pyramid",
    "cone_net": "cone"

}

# Expanded list of Matplotlib-compatible colors
colors = [
    "red", "blue", "green", "yellow", "purple",
    "orange", "cyan", "magenta", "lime", "pink",
    "teal", "lavender", "brown", "beige", "maroon",
    "olive", "navy", "grey", "violet", "turquoise"
]


# Function to plot a 2D net (simplified as a collection of polygons)
def plot_net(ax, net_name, color):
    ax.set_title("Original Shape")
    ax.set_aspect("equal")
    ax.axis("off")
    
    if net_name == "cube_net":
        squares = [
            ([0, 1, 1, 0], [0, 0, 1, 1]),
            ([0, 1, 1, 0], [1, 1, 2, 2]),
            ([0, 1, 1, 0], [2, 2, 3, 3]),
            ([0, 1, 1, 0], [3, 3, 4, 4]),
            ([-1, 0, 0, -1], [2, 2, 3, 3]),
            ([1, 2, 2, 1], [2, 2, 3, 3])
        ]
        for x, y in squares:
            ax.fill(x, y, color=color, alpha=0.5,edgecolor="black")
    elif net_name == "tetrahedron_net":
        tri = np.array([[0, 1, 0.5], [0, 0, np.sqrt(3)/2]])
        ax.fill(tri[0], tri[1], color=color, alpha=0.5,edgecolor="black")
        ax.fill(tri[0]+1, tri[1], color=color, alpha=0.5,edgecolor="black")
        ax.fill(tri[0]+0.5, tri[1]+np.sqrt(3)/2, color=color, alpha=0.5,edgecolor="black")
        ax.fill(tri[0]+1.5, tri[1]+np.sqrt(3)/2, color=color, alpha=0.5,edgecolor="black")
    elif net_name == "octahedron_net":
        tri = np.array([[0, 1, 0.5], [0, 0, np.sqrt(3)/2]])
        for i in range(4):
            ax.fill(tri[0]+i, tri[1], color=color, alpha=0.5,edgecolor="black")
            ax.fill(tri[0]+i, tri[1]+np.sqrt(3)/2, color=color, alpha=0.5,edgecolor="black")
    elif net_name == "dodecahedron_net":
        for i in range(12):
            theta = np.linspace(0, 2*np.pi, 6) + i*2*np.pi/12
            x = 0.5 * np.cos(theta)
            y = 0.5 * np.sin(theta)
            ax.fill(x+i%4, y+(i//4)*1.5, color=color, alpha=0.5,edgecolor="black")
    elif net_name == "icosahedron_net":
        tri = np.array([[0, 0.5, 1], [0, np.sqrt(3)/2, 0]])
        for i in range(5):
            for j in range(4):
                ax.fill(tri[0]+i*0.5, tri[1]+j*np.sqrt(3)/2, color=color, alpha=0.5,edgecolor="black")
    elif net_name == "cuboid_net":
        rects = [
            ([0, 2, 2, 0], [0, 0, 1, 1]),
            ([0, 2, 2, 0], [1, 1, 2, 2]),
            ([0, 2, 2, 0], [2, 2, 3, 3]),
            ([0, 2, 2, 0], [3, 3, 4, 4]),
            ([-1, 0, 0, -1], [2, 2, 3, 3]),
            ([2, 3, 3, 2], [2, 2, 3, 3])
        ]
        for x, y in rects:
            ax.fill(x, y, color=color, alpha=0.5,edgecolor="black")
    elif net_name == "triangular_prism_net":
        # Rectangle 1 (top rectangle)
        ax.fill([0, 1, 1, 0], [2, 2, 3, 3], color=color, alpha=0.5, edgecolor="black")
        # Rectangle 2 (middle rectangle)
        ax.fill([0, 1, 1, 0], [1, 1, 2, 2], color=color, alpha=0.5, edgecolor="black")
        # Rectangle 3 (bottom rectangle)
        ax.fill([0, 1, 1, 0], [0, 0, 1, 1], color=color, alpha=0.5, edgecolor="black")
        # Triangle base 1 (attached to the left side of the middle rectangle)
        ax.fill([0, -np.sqrt(3)/2, 0], [1, 1.5, 2], color=color, alpha=0.5, edgecolor="black")
        # Triangle base 2 (attached to the right side of the middle rectangle)
        ax.fill([1, 1 + np.sqrt(3)/2, 1], [1, 1.5, 2], color=color, alpha=0.5, edgecolor="black")
    elif net_name == "square_pyramid_net":
        # Central square base (1x1 square centered at (0.5, 0.5))
        ax.fill([0, 1, 1, 0], [0, 0, 1, 1], color=color, alpha=0.5, edgecolor="black")
        # Four isosceles triangles, each with base length 1 and height sqrt(3)/2
        # Bottom triangle (base: (0,0) to (1,0), apex: (0.5, -sqrt(3)/2))
        ax.fill([0, 1, 0.5], [0, 0, -np.sqrt(3)/2], color=color, alpha=0.5, edgecolor="black")
        # Top triangle (base: (0,1) to (1,1), apex: (0.5, 1+sqrt(3)/2))
        ax.fill([0, 1, 0.5], [1, 1, 1+np.sqrt(3)/2], color=color, alpha=0.5, edgecolor="black")
        # Left triangle (base: (0,0) to (0,1), apex: (-sqrt(3)/2, 0.5))
        ax.fill([0, 0, -np.sqrt(3)/2], [0, 1, 0.5], color=color, alpha=0.5, edgecolor="black")
        # Right triangle (base: (1,0) to (1,1), apex: (1+sqrt(3)/2, 0.5))
        ax.fill([1, 1, 1+np.sqrt(3)/2], [0, 1, 0.5], color=color, alpha=0.5, edgecolor="black")
    elif net_name == "cone_net":
        # Circular base
        theta = np.linspace(0, 2*np.pi, 100)
        x_base = np.cos(theta)
        y_base = np.sin(theta)
        ax.fill(x_base, y_base, color=color, alpha=0.5, edgecolor="black")
        # Sector for lateral surface
        theta_sector = np.linspace(0, np.pi, 100)
        x_sector = np.cos(theta_sector) * 2
        y_sector = np.sin(theta_sector) * 2 + 2
        ax.fill(x_sector, y_sector, color=color, alpha=0.5, edgecolor="black")

# Function to plot a solid 3D shape
def plot_shape(ax, shape_name, color,label):
    ax.set_title(f"{label}")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if shape_name == "cube":
        v = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ])
        faces = [
            [v[0], v[1], v[3], v[2]], [v[4], v[5], v[7], v[6]],
            [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
            [v[0], v[2], v[6], v[4]], [v[1], v[3], v[7], v[5]]
        ]
        poly = Poly3DCollection(faces, facecolors=color, alpha=0.7, edgecolors="black")
        ax.add_collection3d(poly)
    elif shape_name == "tetrahedron":
        v = np.array([[0, 0, 1], [0, 1, -1], [-1, -1, -1], [1, -1, -1]])
        faces = [
            [v[0], v[1], v[2]], [v[0], v[1], v[3]],
            [v[0], v[2], v[3]], [v[1], v[2], v[3]]
        ]
        poly = Poly3DCollection(faces, facecolors=color, alpha=0.7, edgecolors="black")
        ax.add_collection3d(poly)
    elif shape_name == "octahedron":
        v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        faces = [
            [v[0], v[2], v[4]], [v[0], v[2], v[5]], [v[0], v[3], v[4]], [v[0], v[3], v[5]],
            [v[1], v[2], v[4]], [v[1], v[2], v[5]], [v[1], v[3], v[4]], [v[1], v[3], v[5]]
        ]
        poly = Poly3DCollection(faces, facecolors=color, alpha=0.7, edgecolors="black")
        ax.add_collection3d(poly)
    elif shape_name == "dodecahedron":
        phi = (1 + np.sqrt(5)) / 2
        v = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1], [1, 1, -1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1],
                      [0, phi, 1/phi], [0, -phi, 1/phi], [0, phi, -1/phi], [0, -phi, -1/phi],
                      [1/phi, 0, phi], [-1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, -phi],
                      [phi, 1/phi, 0], [-phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, -1/phi, 0]])
        faces = [
            [v[0], v[8], v[10], v[4], v[16]], [v[0], v[8], v[13], v[1], v[17]], [v[0], v[12], v[2], v[18], v[16]],
            [v[1], v[8], v[10], v[5], v[17]], [v[1], v[13], v[3], v[19], v[17]], [v[2], v[12], v[9], v[6], v[18]],
            [v[2], v[9], v[11], v[3], v[19]], [v[3], v[13], v[7], v[19], v[15]], [v[4], v[10], v[5], v[14], v[16]],
            [v[4], v[14], v[6], v[18], v[16]], [v[5], v[10], v[11], v[7], v[15]], [v[6], v[11], v[7], v[15], v[14]]
        ]
        poly = Poly3DCollection(faces, facecolors=color, alpha=0.7, edgecolors="black")
        ax.add_collection3d(poly)
    elif shape_name == "icosahedron":
        phi = (1 + np.sqrt(5)) / 2
        v = np.array([[0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
                      [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
                      [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]])
        faces = [
            [v[0], v[4], v[8]], [v[0], v[4], v[9]], [v[0], v[5], v[9]], [v[0], v[5], v[8]],
            [v[1], v[4], v[10]], [v[1], v[4], v[11]], [v[1], v[5], v[10]], [v[1], v[5], v[11]],
            [v[2], v[6], v[8]], [v[2], v[6], v[9]], [v[2], v[7], v[8]], [v[2], v[7], v[9]],
            [v[3], v[6], v[10]], [v[3], v[6], v[11]], [v[3], v[7], v[10]], [v[3], v[7], v[11]],
            [v[4], v[8], v[10]], [v[4], v[10], v[11]], [v[5], v[9], v[10]], [v[5], v[9], v[11]]
        ]
        poly = Poly3DCollection(faces, facecolors=color, alpha=0.7, edgecolors="black")
        ax.add_collection3d(poly)
    elif shape_name == "cuboid":
        v = np.array([
            [1, 0.5, 1.5], [1, 0.5, -1.5], [1, -0.5, 1.5], [1, -0.5, -1.5],
            [-1, 0.5, 1.5], [-1, 0.5, -1.5], [-1, -0.5, 1.5], [-1, -0.5, -1.5]
        ])
        faces = [
            [v[0], v[1], v[3], v[2]], [v[4], v[5], v[7], v[6]],
            [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
            [v[0], v[2], v[6], v[4]], [v[1], v[3], v[7], v[5]]
        ]
        poly = Poly3DCollection(faces, facecolors=color, alpha=0.7, edgecolors="black")
        ax.add_collection3d(poly)
    elif shape_name == "triangular_prism":
        v = np.array([[0, 0, -1], [1, 0, -1], [0.5, np.sqrt(3)/2, -1], [0, 0, 1], [1, 0, 1], [0.5, np.sqrt(3)/2, 1]])
        faces = [
            [v[0], v[1], v[2]], [v[3], v[4], v[5]],
            [v[0], v[1], v[4], v[3]], [v[1], v[2], v[5], v[4]], [v[2], v[0], v[3], v[5]]
        ]
        poly = Poly3DCollection(faces, facecolors=color, alpha=0.7, edgecolors="black")
        ax.add_collection3d(poly)
    elif shape_name == "square_pyramid":
        v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]])
        faces = [
            [v[0], v[1], v[2], v[3]], [v[0], v[1], v[4]],
            [v[1], v[2], v[4]], [v[2], v[3], v[4]], [v[3], v[0], v[4]]
        ]
        poly = Poly3DCollection(faces, facecolors=color, alpha=0.7, edgecolors="black")
        ax.add_collection3d(poly)
    elif shape_name == "cone":
        # Create a cone using triangulation
        n = 20  # Number of triangles for approximation
        theta = np.linspace(0, 2*np.pi, n)
        base = np.array([[np.cos(t), np.sin(t), 0] for t in theta])
        apex = np.array([0, 0, 1])
        faces = []
        for i in range(n):
            faces.append([base[i], base[(i+1)%n], apex])  # Lateral faces
        faces.append([base[i] for i in range(n)])  # Base face
        poly = Poly3DCollection(faces, facecolors=color, alpha=0.7, edgecolors="black")
        ax.add_collection3d(poly)


def random_filename(length=5):
    number_part = ''.join(random.choices(string.digits, k=length))
    return number_part + ".png"

# Run the loop 1000 times
if __name__ == "__main__":
    output_dir = r"D:\Perception_dataset\eval\mental_composition_data"
    os.makedirs(output_dir, exist_ok=True)
    annotations = {}
    net_color = None
    shape_colors = None

    # Run the loop 1000 times
    for i in range(100):
        difficulty_level = None
        fig = plt.figure(figsize=(15, 8))
        if i < 50:
            colors = ["white"]
            net_to_shape = easy_net_to_shape
            difficulty_level = "easy"
        else:
            colors = [
                "red", "blue", "green", "yellow", "purple",
                "orange", "cyan", "magenta", "lime", "pink",
                "teal", "lavender", "brown", "beige", "maroon",
                "olive", "navy", "grey", "violet", "turquoise"
            ]
            net_to_shape = hard_net_to_shape
            difficulty_level = "hard"

        all_shapes = list(net_to_shape.values())
        selected_net = random.choice(list(net_to_shape.keys()))
        if len(colors) > 1:
            net_color = random.choice(colors)
            shape_colors = random.sample([color for color in colors if color != net_color],k=3)
        else:
            net_color = random.choice(colors)
            shape_colors = random.sample(colors,k=1)


        correct_shape = net_to_shape[selected_net]
        other_shapes = random.sample([shape for shape in all_shapes if shape != correct_shape], k=3)        
        # Create a list of shapes with their colors
        shape_list = [
            {"shape": correct_shape, "color": random.choice(shape_colors)},
            {"shape": other_shapes[0], "color": random.choice(shape_colors)},
            {"shape": other_shapes[1], "color": net_color},
            {"shape": other_shapes[2], "color": random.choice(shape_colors)}
        ]
        
        # Shuffle the shape list to randomize positions
        random.shuffle(shape_list)
        
        # Assign fixed labels 'a', 'b', 'c' to positions (left to right)
        shape_list[0]["label"] = "a"
        shape_list[1]["label"] = "b"
        shape_list[2]["label"] = "c"
        shape_list[3]["label"] = "d"

        correct_label = next(item["label"] for item in shape_list if item["shape"] == correct_shape)
    
        try:
            ax_net = fig.add_subplot(241)
            plot_net(ax_net, selected_net, net_color)
            
            # Plot shapes in their shuffled positions
            ax_shape1 = fig.add_subplot(245, projection="3d")
            plot_shape(ax_shape1, shape_list[0]["shape"], shape_list[0]["color"], shape_list[0]["label"])
            
            ax_shape2 = fig.add_subplot(246, projection="3d")
            plot_shape(ax_shape2, shape_list[1]["shape"], shape_list[1]["color"], shape_list[1]["label"])
            
            ax_shape3 = fig.add_subplot(247, projection="3d")
            plot_shape(ax_shape3, shape_list[2]["shape"], shape_list[2]["color"], shape_list[2]["label"])
            
            ax_shape2 = fig.add_subplot(248, projection="3d")
            plot_shape(ax_shape2, shape_list[3]["shape"], shape_list[3]["color"], shape_list[3]["label"])
            

            plt.tight_layout()
            filename = random_filename()
            plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches="tight")
            annotations[filename] = {
                "question":"When the 2d figure is folded into a 3d shape, which of the following is the correct shape?",
                "net": selected_net,
                "correct_shape": correct_shape,
                "other_shapes": other_shapes,
                "net_color": net_color,
                "answer": correct_label,
                "difficulty":difficulty_level
            }
            plt.close()
        except Exception as e:
            print(str(e))

    metadata_path = os.path.join(output_dir, "annotation.json")
    print(f"Annotations saved to {metadata_path}")
    with open(metadata_path, "a", encoding="utf-8") as metaf:
        metaf.write(json.dumps(annotations) + "\n")