from lib.slippage import *
from lib.abstract_reasoning import *
from lib.mental_rotation_v2 import generate_one_image
from lib.mental_composition_v2 import *
from lib.paper_folding_v2 import generate_test_image
from lib.dynamic_isomorphism import *
from lib.symmetric_structres import *
from lib.hierarchial import *
import os

def random_filename(label="", length=5):
    number_part = ''.join(random.choices(string.digits, k=length))
    return number_part + "_" + label + ".png"


def generate_slippage(output_dir,num_images):

    solution_hint = {
        "spacing": """To solve this question about transformation puzzle, observe how the spaces in each figure change.""",
        "alignment": """To solve this question about transformation puzzle, observe how the alignment in each figure change.""",
        "number": """To solve this question about transformation puzzle, observe how the number of each object in each figure change.""",
        "enclosure": """To solve this question about transformation puzzle, observe how the inner objects are enclosed bu an outer object in each figure.""",
        "symmetry": """To solve this question about transformation puzzle, observe how the symmetry in each figure change.""",
        "word_symmetry": """To solve this question about transformation puzzle, observe if the words in each figure are the mirror image transformation.""",
        "hollowness": """To solve this question about transformation puzzle, observe if the objects in each figure is hollow ro filled.""",
        "topology": """To solve this question about transformation puzzle, observe how the topology in each figure change.""",
        "border": """To solve this question about transformation puzzle, observe how the border thivkness of each figure in each figure change."""
    }

    slippage_question = """Analyze the six figures labeled A, B, C, D, E, and F in the image.Your goal is to identify the underlying visual concept that is shared by the majority of these figures and then determine which figure does not adhere to this concept.
    """
    word_symmetry_question = """Analyze the six figures labeled A, B, C, D, E, and F in the image. Your goal is to identify the underlying visual concept that is shared by the majority of these figures and then determine which figure does not adhere to this concept. The visual concept is applied to the figure as a whole and not to the parts of the figure."""

    output_dir = os.path.join(output_dir,"slippage") 
    os.makedirs(output_dir, exist_ok=True)
    ### --- Concept Registry ---
    concepts = {
        "spacing": draw_spacing,
        "alignment": draw_alignment,
        "number": draw_number,
        "enclosure": draw_enclosure,
        "symmetry": draw_symmetry,
        "word_symmetry": draw_word_symmetry,
        "hollowness": draw_hollowness,
        "topology": draw_topology,
        "border": draw_border
    }

    option_dict={0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f'}

    random.seed(42)
    chosen_concepts = list(concepts.keys())
    annotations = {}
    for concept in chosen_concepts:
        for img_idx in range(num_images):
            fig, axs = plt.subplots(2, 3, figsize=(25, 10))
            axs = axs.ravel()
            conform_id = random.randint(1,4)
            word_length = random.randint(4, 7)
            word = ''.join(random.choices(string.ascii_uppercase, k=word_length))
            for i in range(6):
                variation_id = random.randint(0, 100)
                ax = axs[i]
                ax.axis('off')
                ax.set_aspect('equal')

                # Add letter label in top-left corner
                label = chr(ord('a') + i)
                ax.text(0.05, 0.95, f'({label})', transform=ax.transAxes,
                        fontsize=15, va='top', ha='left')

                if i == conform_id:
                    if concept == "word_symmetry":
                        concepts[concept](ax, conform=False, word=word)
                    else:
                        concepts[concept](ax, conform=False)
                else:
                    if concept == "word_symmetry":
                        concepts[concept](ax, conform=True)
                    else:
                        concepts[concept](ax, conform=True, variation=variation_id)

            # plt.suptitle(f"{concept.upper()} (Last panel violates)", fontsize=14)

            # Generate random filename
            filename = random_filename(concept)
            filepath = os.path.join(output_dir, filename)
            print(f"Saving figure to {filepath}")
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.3, hspace=0.4)
            plt.savefig(filepath, dpi=150,bbox_inches='tight')
            plt.close()
            print(f"Saved: {filepath}")
            if concept == "word_symmetry":
                annotations[filename]= {
                "question": word_symmetry_question + solution_hint[concept],
                "concept": F"Figure in option {option_dict[conform_id]} violates the concept of {concept}",
                "violation": option_dict[conform_id],
                "answer": word
            }
            else:
                annotations[filename]= {
                        "question": slippage_question + solution_hint[concept],
                        "concept": F"Figure in option {option_dict[conform_id]} violates the concept of {concept}",
                        "violation": option_dict[conform_id],
                    }

    metadata_path = os.path.join(output_dir, "annotation.json")
    with open(metadata_path, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


def generate_abstract_reasoning(dataset_dir,num_image):
    dataset_dir = os.path.join(dataset_dir,"abstract") 
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(dataset_dir,"tmp"),exist_ok=True)
    question = """Analyze the six figures labeled A, B, C, D, E, and F in the image. Your goal is to identify the underlying visual concept that is shared by the majority of these figures and then determine which figure does not adhere to this concept."""
    annotations = {}
    for i in range(num_image):
        try:
            abs_problem_file_name = random_filename("abstract_problem", length=8)
            answer,reason,img_name = gen_abstrct_problems(ntasks=5,abs_problem_file_namesave_dir=os.path.join(dataset_dir,"tmp"),
                                dataset_dir=dataset_dir,process_id=0, random_state=np.random.RandomState(213))
            
            reason = reason.split("-")
            reason1 = reason[0].split("_")[1:]
            reason2 = reason[1].split("_")[:-1]
            completed_reason = f"Other figures {" ".join(reason1)} and {" ".join(reason2)} this one doesn't"
            print(f"Answer: {answer}, Reason: {completed_reason}")
            annotations[img_name] = {
                "question":question,
                "answer": answer,
                "reason": completed_reason
            }
        except Exception as e:
            print(f"Error occurred: {e}")
            continue
    metadata_path = os.path.join(dataset_dir, "annotations.json")
    with open(metadata_path, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


def generate_mrt(dataset_dir,num_images, easy_ratio = 0.5):
    question = """Analyze the four figures labeled A, B, C, D in the image. Your goal is to identify which of these options is the correct rotational transformation fo the figure labeled original shape. Observe the given figures and pay attention to how the original shape can be rotated to match the given options.
    """
    dataset_dir = os.path.join(dataset_dir,"mental_rotation") 
    os.makedirs(dataset_dir, exist_ok=True)
    np.random.seed(42)
    random.seed(42)
    # Generate images
    annotations = {}
    for i in range(num_images):
        if i < int(num_images*easy_ratio):
            single_annotation = generate_one_image(i, difficulty="easy", outdir=dataset_dir)
        else:
            single_annotation = generate_one_image(i, difficulty="complex", outdir=dataset_dir)

        single_annotation["question"] = question,

        annotations[single_annotation["filename"]] = single_annotation

    with open(os.path.join(dataset_dir, "annotations.json"), "a") as f:
        f.write(json.dumps(annotations) + "\n")

    print(f"Generated MRT images in {dataset_dir}")


def generate_mental_composition(output_dir,num_images,easy_ratio = 0.5):
    question = """Analyze the four figures labeled A, B, C, D in the image. These are 3D Figures. Your goal is to identify which of these options can be constructed by folding the top 2D figure labeled as Original Shape. Observe the given figures and pay attention to how the original shape can be folded or composed to match the given options.
    """ 
    output_dir = os.path.join(output_dir,"mental_composition")
    os.makedirs(output_dir, exist_ok=True)
    annotations = {}
    net_color = None
    shape_colors = None

    # Run the loop 1000 times
    for i in range(num_images):
        difficulty_level = None
        fig = plt.figure(figsize=(15, 8))
        if i < int(num_images*easy_ratio):
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
                "question":question,
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

    metadata_path = os.path.join(output_dir, "annotations.json")
    print(f"Annotations saved to {metadata_path}")
    with open(metadata_path, "a", encoding="utf-8") as metaf:
        metaf.write(json.dumps(annotations) + "\n")


def generate_paper_folding(OUTPUT_DIR,num_images):
    question = """The a paper of shape labeled Unfolded is folded follwoing the pattern shown in the top row, in the sequence of fold 1, followed by fold 2 and so on. After the last fold a hole is punched as shown in the image of labeled final view, then after unfolding how would the paper look. Analyze the four figures labeled A, B, C, D in the image, how they are folded and answer. 
    """

    random.seed(42)
    OUTPUT_DIR = os.path.join(OUTPUT_DIR,"paper_folding")
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    metadata_path = os.path.join(OUTPUT_DIR, "annotations.json")
    fold_group_map = {"VH": "Vertical/Horizontal", "Diagonal": "Diagonal"}
    annotations = {}
    with open(metadata_path, "a", encoding="utf-8") as metaf:
        for i in range(1,num_images + 1):
            num_folds = random.randint(2, 4)
            num_holes = 1
            if num_folds > 2:
                num_holes = 1
            else:
                num_holes = random.randint(2, 3)
            fold_group = random.choice(["VH", "Diagonal"])
            if fold_group == "VH":
                folds = [random.choice(["V", "H"]) for _ in range(num_folds)]
            else:
                folds = [random.choice(["D", "N"]) for _ in range(num_folds)]
            paper_size = random.choice([4,6])  # Number of sides for polygon
            image_path, correct_option = generate_test_image(
                folds, paper_size,i, num_folds, num_holes,OUTPUT_DIR
            )
            rel_image_path = os.path.basename(image_path)
            annotations[rel_image_path] = {
                "question":question,
                "correct_option": correct_option,
                "folds": fold_group_map[fold_group],
            }
    with open(metadata_path, "a", encoding="utf-8") as metaf:
        metaf.write(json.dumps(annotations) + "\n")


def generate_dynamic_isomorphism(output_dir,num_images):
    output_dir = os.path.join(output_dir,"dynamic_isomorph")
    os.makedirs(output_dir, exist_ok=True)
    question = """Follow the transformation of the shapes in the top row, starting from t=0.0 uptp t=0.75. Analyze the four figures labeled A, B, C, D in the image. Your goal is to identify which of these options is the fifth image at t = 1.0. Consider how the elements of the labeled figure transforms over time. Following that transformation identify which of these options is the fifth image at t = 1.0
    """
    # Run and generate visuals
    shapes = ["triangle", "diamond", "square", "pentagon", "hexagon"]
    keys = list(transform_map.keys())
    t_list = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # Ensure enough t values
    annotation = {}
    for i in range(num_images):
        # Select two non-overlapping transformations
        (name1, name2) = random.sample(keys, k=2)
        shape1name = random.choice(shapes)
        shape2name = random.choice([shape for shape in shapes if shape != shape1name])
        shape1 = generate_structure(shape1name)
        shape2 = generate_structure(shape2name)
        
        func1 = transform_map[name1]
        func2 = transform_map[name2]
        
        # Save the 2x4 grid
        fname = os.path.join(output_dir, f"pair_{i:02d}_{name1}_{name2}_grid.png")
        fifth_label = plot_grid(shape1, shape2, t_list, func1, func2, fname)
        annotation[f"pair_{i:02d}_{name1}_{name2}_grid.png"] = {
            "question": question,
            "fifth_label": fifth_label,
            "answer":f"The {shape1name} hs the motion of {name1} and the {shape2name} has the motion of {name2}"
        }

    # Save the annotations to a text file
    annotation_file = os.path.join(output_dir, "annotations.json")
    import json
    with open(annotation_file, 'w') as f:
        json.dump(annotation, f, indent=4)

    print("All image grids saved in:", output_dir)
    

def generate_symmetric_structures(output_dir,num_images):
    question = """Analyze the four figures labeled A, B, C, D in the image. Your goal is to identify the underlying visual concept that is shared by the majority of these figures and then determine which figure does not adhere to this concept. Observe the given figures and pay attention to their symmetry properties.
    """
    output_dir = os.path.join(output_dir,"symmetric_isomorph")
    os.makedirs(output_dir,exist_ok=True)
    annotation = {}
    for i in range(num_images):
        asymmetric_label, random_filename = generate_structured_drawing_grid(output_dir)
        annotation[random_filename] = {
            "question": question,
            "asymmetric_label": asymmetric_label,
            "answer":"lacks symmetry"
        }
    # Save the annotations to a text file
    annotation_file = os.path.join(output_dir, "annotations.json")
    with open(annotation_file, 'w') as f:
        import json
        json.dump(annotation, f, indent=4)


def generate_hierarchial_structures(output_dir,num_images):
    question = """Analyze the four figures labeled A, B, C and D in the image. Your goal is to identify the underlying visual concept that is shared by the majority of these figures and then determine which figure does not adhere to this concept. Observe the given figures and pay attention to their recursive structure properties
    """
    output_dir = os.path.join(output_dir,"hierarchial_isomorph")
    os.makedirs(output_dir,exist_ok=True)
    option_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    annotation = {}
    debug_log = []  # To track which functions produce blank figures

    for img_index in range(num_images):
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        axs = axs.flatten()
        
        # Select 3 unique hierarchical functions
        valid_funcs = random.sample(hierarchical_funcs, 3)
        # Pick a random position for the violating structure
        violation_index = random.randint(0, 3)

        for i in range(4):
            axs[i].set_title(f"{option_map[i]}", fontsize=20)
            axs[i].axis('off')
            axs[i].set_aspect('equal')
            axs[i].set_xlim(0, 1)
            axs[i].set_ylim(0, 1)

            # Store current number of patches/lines to detect blank figures
            initial_patches = len(axs[i].patches)
            initial_lines = len(axs[i].lines)

            if i == violation_index:
                violate_structure(axs[i], seed=img_index * 10 + i)
            else:
                func = valid_funcs.pop()
                func(axs[i], seed=img_index * 10 + i)

            # Check if the axes is empty
            if len(axs[i].patches) == initial_patches and len(axs[i].lines) == initial_lines:
                debug_log.append(f"Image {img_index}, Option {option_map[i]}: {func.__name__} produced no visible output")

        plt.tight_layout()
        random_filename = generate_random_filename()
        plt.savefig(os.path.join(output_dir, random_filename))
        plt.close()
        annotation[random_filename] = {
            "question": question,
            "answer": option_map[violation_index]
        }

    # Save the annotation to a JSON file
    annotation_file = os.path.join(output_dir, "annotations.json")
    with open(annotation_file, 'w') as f:
        json.dump(annotation, f, indent=4)


if __name__ == "__main__":
    Dataset_dir = r"D:\Perception_dataset\Data_hint"
    os.makedirs(Dataset_dir,exist_ok=True)
    num_images = 100
    generate_slippage(Dataset_dir,num_images)
    # gen_abstrct_problems(Dataset_dir,num_images*5)
    generate_mrt(Dataset_dir,num_images)
    generate_mental_composition(Dataset_dir,num_images)
    generate_paper_folding(Dataset_dir,num_images)
    generate_dynamic_isomorphism(Dataset_dir,num_images)
    generate_symmetric_structures(Dataset_dir,num_images)
    generate_hierarchial_structures(Dataset_dir,num_images)
