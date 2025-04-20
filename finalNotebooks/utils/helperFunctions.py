
import os
from PIL import Image
import shutil
import numpy as np

def is_rgb(image_path, threshold=2):  # Lower = stricter, 2 is safe
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