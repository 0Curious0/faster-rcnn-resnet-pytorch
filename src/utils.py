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
        x_c = obj["bndbox"]["x_c"]*img_width
        y_c = obj["bndbox"]["y_c"]*img_height
        width = obj["bndbox"]["width"]*img_width
        height = obj["bndbox"]["height"]*img_height

        xmin = x_c - (width / 2)
        ymin = y_c - (height / 2)

        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        ax.text(xmin, ymin - 5, obj["name"], color='r', fontsize=12, weight='bold')

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
        img_array, annotation_dict = dataset[idx]
        
        plot_image_with_annotations(img_array.permute(1, 2, 0).numpy(), annotation_dict)
        

