import importlib 
import bongard
importlib.reload(bongard)
from bongard import BongardProblemPainter
from bongard.util_funcs import get_attribute_sampling_candidates
from bongard.sampler import AbstractSampler
import numpy as np
import os, time
import json
import matplotlib.pyplot as plt
import random
import string


def gen_abstrct_problems(ntasks,abs_problem_file_name,save_dir='', dataset_dir="",process_id=0,
                         random_state=np.random.RandomState(123)):
    shape_actions_filepath = os.getenv("BONGARD_SHAPES_PATH")
    shape_attributes_filepath = os.getenv("BONGARD_ATTRIBUTES_PATH")
    if not shape_actions_filepath or not shape_attributes_filepath:
        raise ValueError(
            "Please set BONGARD_SHAPES_PATH and BONGARD_ATTRIBUTES_PATH "
            "environment variables before generating abstract reasoning data."
        )
    shape_attributes_dict = get_attribute_sampling_candidates(shape_attributes_filepath)

    attribute_list = list(shape_attributes_dict.keys())
    num_attributes = len(attribute_list)

    bongard_attributes = []
    for i in range(num_attributes):
        bongard_attributes.append([attribute_list[i]])
    for i in range(num_attributes):
        for j in range(i + 1, num_attributes):
            bongard_attributes.append([attribute_list[i], attribute_list[j]])

    print('total number of bongard_attributes: ', len(bongard_attributes))

    save_abs_dir = os.path.join(save_dir, 'hd')
    os.makedirs(save_abs_dir, exist_ok=True)

    abstract_sampler = AbstractSampler(shape_actions_filepath, shape_attributes_filepath,
                                       num_positive_examples=7, num_negative_examples=7,
                                       random_state=random_state)
    problem_painter = BongardProblemPainter(scaling_factors_range=(150, 320), magnifier_effect=True, random_seed=0)

    bongard_attributes_item = random.choice(bongard_attributes)
    task_id = random.randint(0,5)
    bongard_abs_problem = abstract_sampler.sample(bongard_attributes_item, task_id)

    if bongard_abs_problem is None:
        print(f"No samples available for attributes: {1} - {bongard_attributes_item}")
        return

    abs_problem_name = bongard_abs_problem.get_problem_name()
    action_program = bongard_abs_problem.get_action_string_list()
    p,n = problem_painter.get_created_bongard_problem(
        bongard_abs_problem,
        bongard_problem_ps_dir=os.path.join(save_abs_dir, 'ps', abs_problem_name),
        bongard_problem_png_dir=os.path.join(save_abs_dir, 'png', abs_problem_name))
        
    print(f"Generated Bongard problem: {abs_problem_name} with {len(p)} positive and {len(n)} negative samples.")
    print(np.array(p[0]).shape, np.array(n[0]).shape)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    images = p[:5] + n[:1]
    image_labels = [chr(ord('A') + i) for i in range(6)]
    # Track which index was the negative image (i.e., index 5 in the list before shuffling)
    original_indices = list(range(6))
    shuffled_indices = original_indices.copy()
    random_state.shuffle(shuffled_indices)


    for plotidx, ax in enumerate(axes.flat):
        idx = shuffled_indices[plotidx]
        arr = np.array(images[idx])
        arr = np.squeeze(arr)
        ax.imshow(arr)
        # ax.set_title(titles[idx])
        ax.set_title(image_labels[plotidx],fontsize=16)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_dir, f"{abs_problem_file_name}"))
    plt.close(fig)
    # Find the label of the negative image
    neg_image_original_index = 5  # The last one added to selected_images
    neg_image_new_pos = shuffled_indices.index(neg_image_original_index)
    neg_image_label = image_labels[neg_image_new_pos]

    print(f"The negative image is at label: {neg_image_label}")
    return neg_image_label,abs_problem_name,abs_problem_file_name
