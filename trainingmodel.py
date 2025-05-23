# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:39:23 2025

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Corrected Cervical Cancer Classification Model

# Check and install required packages
import subprocess
import sys

def install_package(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = [
    "opencv-python",
    "numpy",
    "matplotlib",
    "scikit-learn",
    "tensorflow",
    "seaborn",
    "kagglehub"
]

# Install missing packages
for package in required_packages:
    try:
        __import__(package.replace("-", "_").split(">=")[0])
    except ImportError:
        install_package(package)
        
print("All required packages installed.")
"""

import os
import cv2  # This is from opencv-python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns
import kagglehub

# Step 1: Download the dataset using kagglehub
try:
    dataset_path = kagglehub.dataset_download("prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed")
    print("Dataset downloaded to:", dataset_path)
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Please ensure you have kagglehub installed and are authenticated")
    print("You may need to run: pip install kagglehub")
    print("And then: kaggle authenticate")
    dataset_path = "path/to/your/dataset"  # Fallback path if download fails

# Parameters
input_shape = (224, 224, 3)  # Changed to 3 channels for initial image loading
num_classes = 5
lr = 0.0001  # Slightly increased learning rate
bs = 32  # Reduced batch size to help with memory
ep = 50  # Reduced epochs, will use early stopping instead

# Define paths
base_dir = dataset_path
class_names = ["im_Dyskeratotic", "im_Metaplastic", "im_Koilocytotic", "im_Parabasal", "im_Superficial-Intermediate"]
cropped_dirs = {
    cls: os.path.join(base_dir, cls, cls, "CROPPED")
    for cls in class_names
}
uncropped_dirs = {
    cls: os.path.join(base_dir, cls, cls)
    for cls in class_names
}

# Function to apply Sobel filter
def apply_sobel_filter(image):
    # Convert to grayscale if the image is RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    # Ensure image is in uint8 format
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    
    # Apply Sobel filters
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate magnitude
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize to 0-255 range
    sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return sobel_magnitude

# Load and preprocess images with error handling
def load_images_from_directory(directory, target_size=(224, 224), max_images=None):
    images = []
    file_paths = []
    
    try:
        file_list = os.listdir(directory)
        if max_images:
            file_list = file_list[:max_images]
            
        for filename in file_list:
            img_path = os.path.join(directory, filename)
            if os.path.isfile(img_path):
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, target_size)
                        images.append(img)
                        file_paths.append(img_path)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    except Exception as e:
        print(f"Error accessing directory {directory}: {e}")
        
    return np.array(images) if images else np.array([])

def preprocess_images(images):
    processed_images = []
    for img in images:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply Sobel filter to get edge information
        sobel = apply_sobel_filter(img_rgb)
        
        # Create 4-channel image (RGB + Sobel)
        img_4ch = np.dstack((img_rgb, sobel))
        
        # Normalize pixel values to [0,1]
        img_4ch = img_4ch.astype(np.float32) / 255.0
        
        processed_images.append(img_4ch)
    return np.array(processed_images)

# Load limited number of images per class for memory management
max_images_per_class = 500  # Adjust based on your memory capacity

print("Loading and preprocessing cropped images...")
# Load cropped images
cropped_images = {}
cropped_labels = {}
label_to_index = {cls: idx for idx, cls in enumerate(class_names)}

for cls, dir_path in cropped_dirs.items():
    print(f"Processing class: {cls}")
    images = load_images_from_directory(dir_path, max_images=max_images_per_class)
    if len(images) > 0:
        processed_images = preprocess_images(images)
        cropped_images[cls] = processed_images
        cropped_labels[cls] = np.full(len(processed_images), label_to_index[cls])
        print(f"  - Loaded {len(processed_images)} images")
    else:
        print(f"  - No images found in {dir_path}")

# Check if any images were loaded
if not cropped_images:
    raise ValueError("No images were loaded. Please check the dataset paths.")

# Combine all cropped images and labels
X_cropped = np.concatenate(list(cropped_images.values()))
y_cropped = np.concatenate(list(cropped_labels.values()))
y_cropped_categorical = to_categorical(y_cropped, num_classes=num_classes)

print(f"Total dataset size: {X_cropped.shape[0]} images")
print(f"Input shape: {X_cropped.shape[1:]}")

# Update input_shape based on the actual processed images
input_shape = X_cropped.shape[1:]

# Split the data into 80% training and 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_cropped, y_cropped_categorical, test_size=0.2, random_state=42, stratify=y_cropped
)

print(f"Training set: {X_train.shape[0]} images")
print(f"Validation set: {X_val.shape[0]} images")

# Function to plot sample images
def plot_example_images(images, class_name):
    plt.figure(figsize=(15, 8))
    for i, image in enumerate(images[:6]):  # Display up to 6 images
        # Extract RGB and Sobel channels
        rgb_image = image[:, :, :3]
        sobel_image = image[:, :, 3]
        
        plt.subplot(2, 6, i + 1)
        plt.imshow(rgb_image)
        plt.title(f"{class_name}\nOriginal")
        plt.axis('off')
        
        plt.subplot(2, 6, i + 7)
        plt.imshow(sobel_image, cmap='gray')
        plt.title("Sobel Edge")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{class_name}_sample_images.png")
    plt.close()

# Plot sample images from each class
for cls in class_names:
    if cls in cropped_images and len(cropped_images[cls]) > 0:
        plot_example_images(cropped_images[cls], cls)

# Build the model
print("Building model...")
input_layer = Input(shape=input_shape)

# First block
x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)

# Second block
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)

# Third block
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)

# Fourth block
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)

# Global pooling and fully connected layers
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks for better training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]

# Print model summary
model.summary()

# Train the model
print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=ep,
    batch_size=bs,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on validation data
print("Evaluating on validation data...")
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Load and preprocess uncropped images for testing
print("Loading and preprocessing uncropped test images...")
uncropped_images = {}
uncropped_labels = {}

for cls, dir_path in uncropped_dirs.items():
    print(f"Processing uncropped class: {cls}")
    images = load_images_from_directory(dir_path, max_images=max_images_per_class)
    if len(images) > 0:
        processed_images = preprocess_images(images)
        uncropped_images[cls] = processed_images
        uncropped_labels[cls] = np.full(len(processed_images), label_to_index[cls])
        print(f"  - Loaded {len(processed_images)} images")
    else:
        print(f"  - No images found in {dir_path}")

# Check if any uncropped images were loaded
if uncropped_images:
    # Combine all uncropped images and labels
    X_uncropped = np.concatenate(list(uncropped_images.values()))
    y_uncropped = np.concatenate(list(uncropped_labels.values()))
    y_uncropped_categorical = to_categorical(y_uncropped, num_classes=num_classes)
    
    # Evaluate on uncropped data
    print("Evaluating on uncropped test data...")
    uncropped_loss, uncropped_accuracy = model.evaluate(X_uncropped, y_uncropped_categorical)
    print(f"Test Loss (Uncropped): {uncropped_loss:.4f}")
    print(f"Test Accuracy (Uncropped): {uncropped_accuracy:.4f}")
else:
    print("No uncropped images were loaded for testing.")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

# Generate predictions and evaluation metrics for validation set
print("Generating evaluation metrics...")
y_pred_val = model.predict(X_val)
y_pred_classes_val = np.argmax(y_pred_val, axis=1)
y_true_classes_val = np.argmax(y_val, axis=1)

# Classification report
print("Classification Report (Validation Data):")
print(classification_report(y_true_classes_val, y_pred_classes_val, target_names=class_names))

# Confusion matrix for validation data
conf_matrix_val = confusion_matrix(y_true_classes_val, y_pred_classes_val)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Blues',
            xticklabels=[name.replace("im_", "") for name in class_names],
            yticklabels=[name.replace("im_", "") for name in class_names])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Validation Data)')
plt.tight_layout()
plt.savefig("validation_confusion_matrix.png")
plt.show()

# ROC Curve for validation data
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_val[:, i], y_pred_val[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 
             label=f'ROC curve {class_names[i].replace("im_", "")} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multi-class Classification')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curves.png")
plt.show()

# If uncropped images were loaded, generate evaluation metrics for them too
if 'X_uncropped' in locals():
    # Generate predictions for uncropped test data
    y_pred_uncropped = model.predict(X_uncropped)
    y_pred_classes_uncropped = np.argmax(y_pred_uncropped, axis=1)
    y_true_classes_uncropped = np.argmax(y_uncropped_categorical, axis=1)
    
    # Classification report for uncropped data
    print("Classification Report (Uncropped Test Data):")
    print(classification_report(y_true_classes_uncropped, y_pred_classes_uncropped, target_names=class_names))
    
    # Confusion matrix for uncropped test data
    conf_matrix_uncropped = confusion_matrix(y_true_classes_uncropped, y_pred_classes_uncropped)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_uncropped, annot=True, fmt='d', cmap='Greens',
                xticklabels=[name.replace("im_", "") for name in class_names],
                yticklabels=[name.replace("im_", "") for name in class_names])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix (Uncropped Test Data)')
    plt.tight_layout()
    plt.savefig("uncropped_confusion_matrix.png")
    plt.show()

# Save the model
model_save_path = "sipakmed_cervical_model.h5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")