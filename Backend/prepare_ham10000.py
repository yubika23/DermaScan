"""
HAM10000 Dataset Preparation Script
This script organizes HAM10000 images into train/validation/test folders
"""

import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter

# Paths - UPDATE THESE to match your structure
METADATA_CSV = 'dataset/train/archive/HAM10000_metadata.csv'  # Path to your CSV
IMAGES_PART1 = 'dataset/train/archive/HAM10000_images_part_1'  # Path to images part 1
IMAGES_PART2 = 'dataset/train/archive/HAM10000_images_part_2'  # Path to images part 2

OUTPUT_DIR = 'dataset'

# HAM10000 class mapping
CLASS_NAMES = {
    'mel': 'Melanoma',
    'nv': 'Melanocytic_nevus',
    'bcc': 'Basal_cell_carcinoma',
    'akiec': 'Actinic_keratosis',
    'bkl': 'Benign_keratosis',
    'df': 'Dermatofibroma',
    'vasc': 'Vascular_lesion'
}

def create_folders():
    """Create train/validation/test folders for each class"""
    for split in ['train', 'validation', 'test']:
        for class_name in CLASS_NAMES.values():
            folder_path = os.path.join(OUTPUT_DIR, split, class_name)
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created: {folder_path}")

def find_image(image_id, part1_dir, part2_dir):
    """Find image in either part 1 or part 2 folder"""
    # Try .jpg extension in part 1
    path1 = os.path.join(part1_dir, f"{image_id}.jpg")
    if os.path.exists(path1):
        return path1
    
    # Try .jpg extension in part 2
    path2 = os.path.join(part2_dir, f"{image_id}.jpg")
    if os.path.exists(path2):
        return path2
    
    return None

def prepare_dataset():
    """Organize HAM10000 dataset into train/val/test splits"""
    
    print("Loading metadata...")
    df = pd.read_csv(METADATA_CSV)
    
    print(f"Total images: {len(df)}")
    print("\nClass distribution:")
    print(df['dx'].value_counts())
    
    # Create folders
    print("\nCreating folder structure...")
    create_folders()
    
    # Split data: 70% train, 15% validation, 15% test
    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=42, 
        stratify=df['dx']
    )
    
    # Second split: 50% of temp = 15% val, 15% test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['dx']
    )
    
    print(f"\nTrain: {len(train_df)} images")
    print(f"Validation: {len(val_df)} images")
    print(f"Test: {len(test_df)} images")
    
    # Copy files
    datasets = {
        'train': train_df,
        'validation': val_df,
        'test': test_df
    }
    
    for split_name, split_df in datasets.items():
        print(f"\nProcessing {split_name}...")
        copied = 0
        not_found = 0
        
        for idx, row in split_df.iterrows():
            image_id = row['image_id']
            dx = row['dx']  # diagnosis code
            class_name = CLASS_NAMES[dx]
            
            # Find image file
            src_path = find_image(image_id, IMAGES_PART1, IMAGES_PART2)
            
            if src_path:
                # Destination path
                dst_path = os.path.join(OUTPUT_DIR, split_name, class_name, f"{image_id}.jpg")
                
                # Copy file
                shutil.copy2(src_path, dst_path)
                copied += 1
                
                if copied % 100 == 0:
                    print(f"  Copied {copied} images...")
            else:
                not_found += 1
                print(f"  Warning: Image not found: {image_id}")
        
        print(f"  ✓ Copied {copied} images")
        if not_found > 0:
            print(f"  ⚠ {not_found} images not found")
    
    # Print final statistics
    print("\n" + "="*60)
    print("Dataset prepared successfully!")
    print("="*60)
    
    for split in ['train', 'validation', 'test']:
        print(f"\n{split.upper()}:")
        for class_name in CLASS_NAMES.values():
            folder_path = os.path.join(OUTPUT_DIR, split, class_name)
            count = len(os.listdir(folder_path))
            print(f"  {class_name}: {count} images")

if __name__ == '__main__':
    print("="*60)
    print("  HAM10000 Dataset Preparation")
    print("="*60)
    print()
    
    # Check if files exist
    if not os.path.exists(METADATA_CSV):
        print(f"❌ Error: Metadata file not found: {METADATA_CSV}")
        print("\nPlease update the paths in this script:")
        print("  - METADATA_CSV")
        print("  - IMAGES_PART1")
        print("  - IMAGES_PART2")
        exit(1)
    
    if not os.path.exists(IMAGES_PART1):
        print(f"❌ Error: Images folder not found: {IMAGES_PART1}")
        exit(1)
    
    # Run preparation
    response = input("This will organize images into train/val/test folders. Continue? (y/n): ")
    
    if response.lower() == 'y':
        prepare_dataset()
        print("\n✅ Done! You can now train your model.")
    else:
        print("Cancelled.")