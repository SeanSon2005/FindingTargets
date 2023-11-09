import os
import glob
import shutil

base_images = glob.glob('base_images/*')
base_labels = glob.glob('base_labels/*')

TRAINING = 0.85

total_num = len(base_images)
training_num = int(total_num * TRAINING)

# clear previous folders
files = glob.glob('data/train/images/*')
for f in files:
    os.remove(f)
files = glob.glob('data/valid/images/*')
for f in files:
    os.remove(f)
files = glob.glob('data/train/labels/*')
for f in files:
    os.remove(f)
files = glob.glob('data/valid/labels/*')
for f in files:
    os.remove(f)

for i, file in enumerate(base_images):
    if i < training_num:
        shutil.move(file, "data/train/images")
    else:
        shutil.move(file, "data/valid/images")

for i, file in enumerate(base_labels):
    if i < training_num:
        shutil.move(file, "data/train/labels")
    else:
        shutil.move(file, "data/valid/labels")
