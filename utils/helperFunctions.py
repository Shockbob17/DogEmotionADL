
from PIL import Image
import numpy as np
import gdown
import os

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

def unnormalize(tensor, mean, std):
    """
    Reverses the normalization of an image tensor.
    Args:
        tensor (Tensor): Normalized image tensor (C, H, W)
        mean (list or tuple): Mean used in normalization
        std (list or tuple): Std used in normalization
    Returns:
        Tensor: Unnormalized image tensor
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # reverse of (x - mean) / std
    return tensor

def download_model(root_dir: str, model_name: str, zip_url: str, ending_str="_best_model.pt"):
    """
    
    """
    # Creating model directory if doesn't exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        print(F"Creating root directory, {root_dir}")
    
    model_dir = os.path.join(root_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(F"Creating {model_name} directory in {root_dir}")

    model_name = os.path.join(root_dir, model_name, f"{model_name}{ending_str}")
    if not os.path.exists(model_name):
        try:
            gdown.download(zip_url, f"{model_name}{ending_str}", quiet=False)
            print(F"Model for {model_name} downloaded")
        except:
            
    else:

