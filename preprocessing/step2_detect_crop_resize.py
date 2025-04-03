import os
from PIL import Image
from ultralytics import YOLO
import torch

# Directories
input_dir = "datasets/dog_emotion_rgb"
output_dir = "datasets/dog_faces_224"
no_detect_log = "no_dogs_detected.txt"

# Load YOLOv8 model
print("📦 Loading YOLOv8 model...")
model = YOLO("yolov8x.pt")  # Or yolov8n.pt for smaller model
model.to("cuda" if torch.cuda.is_available() else "cpu")
print("✅ YOLOv8 loaded.\n")

# Counters
total = 0
detected = 0
no_dog = 0

# Clear old log if exists
if os.path.exists(no_detect_log):
    os.remove(no_detect_log)

# Start processing
for root, _, files in os.walk(input_dir):
    for file in files:
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(root, file)
        relative_path = os.path.relpath(img_path, input_dir)
        save_path = os.path.join(output_dir, relative_path)

        print(f"📷 Processing: {img_path}")

        try:
            results = model(img_path, conf=0.15)  # lowered threshold
            result = results[0]
            dog_found = False

            for box in result.boxes:
                class_id = int(box.cls)
                if class_id == 16:  # class 16 = dog
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    img = Image.open(img_path).convert("RGB")
                    cropped = img.crop((x1, y1, x2, y2)).resize((224, 224))
                    cropped.save(save_path)

                    print(f"✅ Saved cropped dog to {save_path}")
                    detected += 1
                    dog_found = True
                    break

            if not dog_found:
                with open(no_detect_log, "a") as f:
                    f.write(f"{img_path}\n")
                print("🐶 No dog class (16) detected.")
                no_dog += 1

        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")
        
        total += 1

# Summary
print("\n📊 Step 2 Summary:")
print(f"Total images processed : {total}")
print(f"Dog detections         : {detected}")
print(f"No detections          : {no_dog}")
print(f"Saved to folder        : {output_dir}")
print(f"No-dog image log       : {no_detect_log}")
