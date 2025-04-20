
from PIL import Image
import numpy as np

def is_rgb(image_path, threshold=2):  # Lower = stricter, 2 is safe
    """
    Checks whether an image is RGB (color) or grayscale based on pixel color variance.

    This function opens an image, converts it to RGB, and then checks the standard deviation of the color channels 
    across all pixels. If the standard deviation is low (below a specified threshold), the image is considered grayscale.

    Args:
        image_path (str): Path to the image file to be checked.
        threshold (int, optional): A threshold value for the mean standard deviation across color channels. 
                                   If the mean standard deviation is lower than this value, the image is considered grayscale.
                                   Default is 2, which is a safe value for most images.

    Returns:
        bool: True if the image has color (RGB), False if it is grayscale or nearly grayscale.

    Raises:
        Exception: If there is an issue opening or processing the image, the function will catch and print the error,
                   and will return False, treating the image as non-RGB.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        np_img = np.array(img)

        # Get standard deviation across color channels per pixel
        std_dev = np.std(np_img, axis=2)

        # If std dev is very low, it's grayscale (or very close)
        if np.mean(std_dev) < threshold:
            return False
        return True

    except Exception as e:
        print(f"Skipped {image_path} due to error: {e}")
        return False
    