import os
import shutil
import random

input_folder = 'datasets/dog_faces_augmented/Dog Emotion'
output_base = 'datasets/final_split'

# Splits
split_ratios = {
    'train': 0.7,
    'test': 0.15,
    'eval': 0.15
}

# Seed for reproducibility
random.seed(42)

# Clear and recreate output folders
for split in split_ratios:
    split_path = os.path.join(output_base, split)
    if os.path.exists(split_path):
        shutil.rmtree(split_path)
    os.makedirs(split_path)

total_images = 0
split_counts = {'train': 0, 'test': 0, 'eval': 0}

# For each class folder
for class_folder in os.listdir(input_folder):
    print(f"📂 Found folder: {class_folder}")
    class_path = os.path.join(input_folder, class_folder)
    if not os.path.isdir(class_path):
        continue

    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"🖼️  Found {len(images)} images in {class_folder}")

    random.shuffle(images)

    n = len(images)
    n_train = int(n * split_ratios['train'])
    n_test = int(n * split_ratios['test'])
    n_eval = n - n_train - n_test

    split_lists = {
        'train': images[:n_train],
        'test': images[n_train:n_train + n_test],
        'eval': images[n_train + n_test:]
    }

    for split, split_imgs in split_lists.items():
        for img_name in split_imgs:
            src = os.path.join(class_path, img_name)
            dst_dir = os.path.join(output_base, split, class_folder)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src, os.path.join(dst_dir, img_name))
            split_counts[split] += 1

    total_images += n

# Summary
print("\n📊 Step 4 Summary:")
print(f"Total images processed : {total_images}")
for split in split_ratios:
    print(f"{split.capitalize()} images        : {split_counts[split]}")
print(f"✅ Data split saved in: {output_base}")
