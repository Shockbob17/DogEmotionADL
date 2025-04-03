import os
from PIL import Image
import shutil
import numpy as np

# Input/output folder paths
input_folder = 'datasets/dog_emotion'
output_folder = 'datasets/dog_emotion_rgb'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Counters
total_images = 0
bw_images = 0
copied_images = 0
skipped_images = 0

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
        print(f"⚠️ Skipped {image_path} due to error: {e}")
        return False

# Scan and filter images
for root, _, files in os.walk(input_folder):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            total_images += 1
            path = os.path.join(root, filename)
            if is_rgb(path):
                # Preserve subfolder structure
                relative_path = os.path.relpath(path, input_folder)
                destination = os.path.join(output_folder, relative_path)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copy(path, destination)
                copied_images += 1
            else:
                print(f"Removed (Not RGB): {filename}")
                bw_images += 1

# Skipped due to load errors
skipped_images = total_images - (copied_images + bw_images)

# Summary
print("\n📊 Preprocessing Summary:")
print(f"Total images scanned      : {total_images}")
print(f"Black & white images      : {bw_images}")
print(f"Images removed            : {bw_images + skipped_images}")
print(f"Images successfully kept  : {copied_images}")
print(f"Images skipped (corrupted): {skipped_images}")
print(f"✅ RGB images saved in: {output_folder}")
