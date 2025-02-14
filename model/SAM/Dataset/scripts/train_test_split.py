# Generated by ChatGPT
# Prompt: I have a folder of image, conbined with annotations, in names like A01-1.bmp and A01-1.json.
#   Now I want to shuffle them into Train and Test folder. Suggest python code.
#   I want to ensure the bmp and json file stays together, and the train-test ratio could be configurable, and have a initial number of 0.2

import os
import shutil
import random

def split_dataset(input_folder, train_folder, test_folder, test_ratio=0.2):
    """
    Splits dataset into Train and Test folders.

    Args:
        input_folder (str): Path to the folder containing .bmp and .json files.
        train_folder (str): Path to the output Train folder.
        test_folder (str): Path to the output Test folder.
        test_ratio (float): Ratio of files to go into the Test folder (default: 0.2).
    """
    # Ensure the output directories exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get all .bmp files
    bmp_files = [f for f in os.listdir(input_folder) if f.endswith('.bmp')]

    # Shuffle files
    random.shuffle(bmp_files)

    # Split into train and test sets
    split_idx = int(len(bmp_files) * (1 - test_ratio))
    train_files = bmp_files[:split_idx]
    test_files = bmp_files[split_idx:]

    # Move files to Train folder
    for file_name in train_files:
        base_name = os.path.splitext(file_name)[0]
        bmp_path = os.path.join(input_folder, f"{base_name}.bmp")
        json_path = os.path.join(input_folder, f"{base_name}.json")

        if os.path.exists(bmp_path):
            shutil.copy(bmp_path, train_folder)
        if os.path.exists(json_path):
            shutil.copy(json_path, train_folder)

    # Move files to Test folder
    for file_name in test_files:
        base_name = os.path.splitext(file_name)[0]
        bmp_path = os.path.join(input_folder, f"{base_name}.bmp")
        json_path = os.path.join(input_folder, f"{base_name}.json")

        if os.path.exists(bmp_path):
            shutil.copy(bmp_path, test_folder)
        if os.path.exists(json_path):
            shutil.copy(json_path, test_folder)



script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, "../annotations_cropped")
train_folder = os.path.join(script_dir, "../Train_raw")
test_folder = os.path.join(script_dir, "../Test_raw")

# Set test ratio (default 0.2)
test_ratio = 0.2
split_dataset(input_folder, train_folder, test_folder, test_ratio=test_ratio)
