import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_image_with_annotations(img_array, annotation_dict, img_width=None, img_height=None):
    fig, ax = plt.subplots(1)
    ax.imshow(img_array)

    if img_width is None:
        img_width = annotation_dict["size"]["width"]
    if img_height is None:
        img_height = annotation_dict["size"]["height"]

    for obj in annotation_dict["objects"]:
        x_min = obj["bndbox"]["x_min"]*img_width
        y_min = obj["bndbox"]["y_min"]*img_height
        x_max = obj["bndbox"]["x_max"]*img_width
        y_max = obj["bndbox"]["y_max"]*img_height

        width = x_max - x_min
        height = y_max - y_min

        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        ax.text(x_min, y_min - 5, obj["name"], color='r', fontsize=12, weight='bold')

    plt.show()


def display_random_images_with_annotations(dataset, 
                                           num_images=5,
                                           display_shape : bool = True,
                                           seed : int = None):
    """
    Displays a random selection of images from the dataset along with their annotations.

    Args:
        dataset: A dataset object that provides access to images and their annotations.
        num_images: The number of random images to display.
        display_shape: If True, displays the shape of each image.
        seed: Random seed for reproducibility.
    """

    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    if num_images > 10:
        num_images = 10
        print("Warning: Displaying more than 10 images may clutter the output. Displaying only 10 images.")

    random_indices = random.sample(range(len(dataset)), num_images)

    for idx in random_indices:
        img_tensor, annotation_dict = dataset[idx]
        
        plot_image_with_annotations(img_tensor.permute(1, 2, 0).numpy(), annotation_dict, img_width=img_tensor.shape[2], img_height=img_tensor.shape[1])


