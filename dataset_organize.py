import os
import shutil

# Root dataset folder
source_dir = "breastcancer"  # This folder contains subfolders like 8863, 8864...
output_dir = "dataset_sample"

# Output folders
benign_dir = os.path.join(output_dir, "benign")
malignant_dir = os.path.join(output_dir, "malignant")
os.makedirs(benign_dir, exist_ok=True)
os.makedirs(malignant_dir, exist_ok=True)

# Limits
max_images = 1000
benign_count = 0
malignant_count = 0

# Loop through subfolders (e.g., 8863, 8864, etc.)
for subfolder in os.listdir(source_dir):
    subfolder_path = os.path.join(source_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    # Inside each subfolder, there should be 0/ and 1/
    for label in ["0", "1"]:
        label_path = os.path.join(subfolder_path, label)
        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            if file.endswith(".png"):
                src_file = os.path.join(label_path, file)

                # Copy to benign
                if label == "0" and benign_count < max_images:
                    dst_file = os.path.join(benign_dir, f"{subfolder}_{file}")
                    shutil.copy(src_file, dst_file)
                    benign_count += 1

                # Copy to malignant
                elif label == "1" and malignant_count < max_images:
                    dst_file = os.path.join(malignant_dir, f"{subfolder}_{file}")
                    shutil.copy(src_file, dst_file)
                    malignant_count += 1

                # Stop if both limits reached
                if benign_count >= max_images and malignant_count >= max_images:
                    break
        if benign_count >= max_images and malignant_count >= max_images:
            break
    if benign_count >= max_images and malignant_count >= max_images:
        break

print(f"✅ Done! Copied {benign_count} benign and {malignant_count} malignant images to 'dataset_sample/'")
