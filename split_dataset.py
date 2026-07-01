import os
import shutil
import random

source_dir = "./data"


output_dir = "./data2"

train_ratio = 0.80
val_ratio = 0.10
test_ratio = 0.10

classes = [d for d in os.listdir(
    source_dir) if os.path.isdir(os.path.join(source_dir, d))]

splits = ['train', 'val', 'test']
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

for cls in classes:
    class_path = os.path.join(source_dir, cls)
    all_images = [f for f in os.listdir(
        class_path) if os.path.isfile(os.path.join(class_path, f))]

    random.seed(42)
    random.shuffle(all_images)

    total_imgs = len(all_images)
    train_end = int(total_imgs * train_ratio)
    val_end = train_end + int(total_imgs * val_ratio)

    train_imgs = all_images[:train_end]
    val_imgs = all_images[train_end:val_end]
    test_imgs = all_images[val_end:]

    def copy_files(files, split_name):
        for f in files:
            src = os.path.join(source_dir, cls, f)
            dst = os.path.join(output_dir, split_name, cls, f)
            shutil.copy(src, dst)

    copy_files(train_imgs, 'train')
    copy_files(val_imgs, 'val')
    copy_files(test_imgs, 'test')

print("Folders created successfully! Your data is now ready in the './data' directory.")
