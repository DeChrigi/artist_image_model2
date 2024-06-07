import os
import random
import shutil

def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    
    categories = os.listdir(source_dir)
    
    for category in categories:
        category_path = os.path.join(source_dir, category)
        if os.path.isdir(category_path):
            images = os.listdir(category_path)
            random.shuffle(images)
            
            train_count = int(len(images) * split_ratio)
            
            train_images = images[:train_count]
            val_images = images[train_count:]
            
            category_train_dir = os.path.join(train_dir, category)
            category_val_dir = os.path.join(val_dir, category)
            
            if not os.path.exists(category_train_dir):
                os.makedirs(category_train_dir)
            if not os.path.exists(category_val_dir):
                os.makedirs(category_val_dir)
            
            for image in train_images:
                shutil.copy(os.path.join(category_path, image), os.path.join(category_train_dir, image))
            
            for image in val_images:
                shutil.copy(os.path.join(category_path, image), os.path.join(category_val_dir, image))

source_directory = './dataset/images/images'
train_directory = './dataset/train_images'
val_directory = './dataset/test_images'

split_data(source_directory, train_directory, val_directory)