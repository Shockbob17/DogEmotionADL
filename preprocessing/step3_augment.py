import os
from PIL import Image
from torchvision import transforms
import random

# Input/output folders
input_folder = 'datasets/dog_faces_224'
output_folder = 'datasets/dog_faces_augmented'

# Define augmentation transforms
augmentation_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.Resize((224, 224))
])

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Set number of augmentations per image
NUM_AUGMENTATIONS = 2

total_images = 0
augmented_images = 0

for root, _, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            total_images += 1
            input_path = os.path.join(root, file)

            # Load image
            try:
                img = Image.open(input_path).convert('RGB')
            except Exception as e:
                print(f"⚠️ Skipping {input_path} due to error: {e}")
                continue

            # Reconstruct class subfolder path
            relative_path = os.path.relpath(root, input_folder)
            class_output_dir = os.path.join(output_folder, relative_path)
            os.makedirs(class_output_dir, exist_ok=True)

            # Save original (resized) image
            original_resized = transforms.Resize((224, 224))(img)
            original_resized.save(os.path.join(class_output_dir, file))

            # Generate augmented images
            for i in range(NUM_AUGMENTATIONS):
                augmented = augmentation_pipeline(img)
                aug_filename = f"{os.path.splitext(file)[0]}_aug{i+1}.jpg"
                augmented.save(os.path.join(class_output_dir, aug_filename))
                augmented_images += 1

print("\n📊 Step 3 Summary:")
print(f"Total original images     : {total_images}")
print(f"Augmented images created  : {augmented_images}")
print(f"✅ Augmented data saved in: {output_folder}")
