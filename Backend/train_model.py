"""
Skin Disease Detection Model Training Script
This script trains a CNN model for skin disease classification
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 7  # Adjust based on your dataset

# Dataset paths - UPDATE THESE PATHS
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/validation'
TEST_DIR = 'dataset/test'

# Class names - Update based on your dataset
CLASS_NAMES = [
    'Melanoma',
    'Melanocytic_nevus',
    'Basal_cell_carcinoma',
    'Actinic_keratosis',
    'Benign_keratosis',
    'Dermatofibroma',
    'Vascular_lesion'
]

def create_data_generators():
    """Create data generators with augmentation"""
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation/Test data - only rescaling
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES
    )
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES
    )
    
    return train_generator, val_generator

def create_model_simple():
    """Create a simple CNN model"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # Convolutional blocks
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_model_transfer_learning():
    """Create model using transfer learning (MobileNetV2)"""
    
    # Load pre-trained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create new model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model, base_model

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def train_model_simple():
    """Train simple CNN model"""
    print("Creating data generators...")
    train_gen, val_gen = create_data_generators()
    
    print("Creating model...")
    model = create_model_simple()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'models/skin_disease_model_best.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Save final model
    model.save('models/skin_disease_model.h5')
    print("Model saved to models/skin_disease_model.h5")
    
    # Plot history
    plot_training_history(history)
    
    return model, history

def train_model_transfer_learning():
    """Train model using transfer learning"""
    print("Creating data generators...")
    train_gen, val_gen = create_data_generators()
    
    print("Creating model with transfer learning...")
    model, base_model = create_model_transfer_learning()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'models/skin_disease_model_transfer_best.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Phase 1: Train with frozen base
    print("\n=== Phase 1: Training with frozen base model ===")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks
    )
    
    # Phase 2: Fine-tune some layers
    print("\n=== Phase 2: Fine-tuning ===")
    base_model.trainable = True
    
    # Freeze all layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        callbacks=callbacks
    )
    
    # Save final model
    model.save('models/skin_disease_model_transfer.h5')
    print("Model saved to models/skin_disease_model_transfer.h5")
    
    # Combine histories
    history = history1
    for key in history.history.keys():
        history.history[key].extend(history2.history[key])
    
    # Plot history
    plot_training_history(history)
    
    return model, history

if __name__ == '__main__':
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Choose training method
    print("Choose training method:")
    print("1. Simple CNN (train from scratch)")
    print("2. Transfer Learning (recommended - faster and better)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        model, history = train_model_simple()
    elif choice == '2':
        model, history = train_model_transfer_learning()
    else:
        print("Invalid choice. Using transfer learning by default.")
        model, history = train_model_transfer_learning()
    
    print("\nTraining completed!")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")