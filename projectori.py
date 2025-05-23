import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd

class CancerDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cervical Cancer Detection Class Detection System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")

        # Model parameters
        self.model = None
        self.class_names = ["Benign", "Malignant"]  # Default classes
        self.img_size = (224, 224)
        self.model_trained = False
        
        # Create frames
        self.create_header_frame()
        self.create_main_frame()
        self.create_bottom_frame()
        
    def create_header_frame(self):
        """Create the header with title and description"""
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(header_frame, text="Cervical Cancer Class Detection Detection System", 
                              font=("Arial", 24, "bold"), bg="#2c3e50", fg="white")
        title_label.pack(pady=20)
        
    def create_main_frame(self):
        """Create the main content area"""
        self.main_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel for image display and prediction
        self.left_panel = tk.LabelFrame(self.main_frame, text="Image Preview", bg="#f0f0f0", font=("Arial", 12))
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Display area for the image
        self.image_display = tk.Label(self.left_panel, bg="white", width=40, height=15)
        self.image_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Right panel for results
        self.right_panel = tk.LabelFrame(self.main_frame, text="Results", bg="#f0f0f0", font=("Arial", 12))
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Results text area
        self.result_text = tk.Text(self.right_panel, height=10, width=40, font=("Arial", 10))
        self.result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
    def create_bottom_frame(self):
        """Create the bottom frame with buttons and controls"""
        bottom_frame = tk.Frame(self.root, bg="#f0f0f0", height=150)
        bottom_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Buttons for actions
        self.btn_load_image = tk.Button(bottom_frame, text="Load Image", command=self.load_image,
                               bg="#3498db", fg="white", font=("Arial", 12), width=15)
        self.btn_load_image.grid(row=0, column=0, padx=5, pady=10)
        
        self.btn_predict = tk.Button(bottom_frame, text="Predict", command=self.predict,
                               bg="#2ecc71", fg="white", font=("Arial", 12), width=15)
        self.btn_predict.grid(row=0, column=1, padx=5, pady=10)
        
        self.btn_train_model = tk.Button(bottom_frame, text="Train New Model", command=self.train_model,
                               bg="#e74c3c", fg="white", font=("Arial", 12), width=15)
        self.btn_train_model.grid(row=0, column=2, padx=5, pady=10)
        
        self.btn_load_model = tk.Button(bottom_frame, text="Load Existing Model", command=self.load_existing_model,
                               bg="#9b59b6", fg="white", font=("Arial", 12), width=15)
        self.btn_load_model.grid(row=0, column=3, padx=5, pady=10)
        
        # Configure grid weights
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=1)
        bottom_frame.grid_columnconfigure(2, weight=1)
        bottom_frame.grid_columnconfigure(3, weight=1)
        
    def load_image(self):
        """Load an image from file system"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                # Open and display the image
                self.original_image = Image.open(file_path)
                self.display_image = self.original_image.copy()
                
                # Resize for display while maintaining aspect ratio
                display_img = self.resize_image_for_display(self.display_image)
                self.photo_image = ImageTk.PhotoImage(display_img)
                
                self.image_display.config(image=self.photo_image)
                self.image_path = file_path
                
                # Clear previous results
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Image loaded: {os.path.basename(file_path)}\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def resize_image_for_display(self, image, max_size=(350, 350)):
        """Resize image for display while maintaining aspect ratio"""
        width, height = image.size
        ratio = min(max_size[0]/width, max_size[1]/height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.LANCZOS)
    
    def predict(self):
        """Make prediction on the loaded image"""
        if not hasattr(self, 'original_image'):
            messagebox.showinfo("Info", "Please load an image first.")
            return
            
        if not self.model_trained and self.model is None:
            messagebox.showinfo("Info", "Please train or load a model first.")
            return
            
        try:
            # Preprocess the image
            img = self.original_image.resize(self.img_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = self.model.predict(img_array)
            
            # Display results
            self.display_results(prediction)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction: {str(e)}")
    
    def display_results(self, prediction):
        """Display the prediction results"""
        # Clear previous results
        self.result_text.delete(1.0, tk.END)
        
        # Assuming binary classification (cancer/no cancer)
        if len(prediction[0]) == 1:  # Binary classification
            probability = prediction[0][0]
            result = "Positive (Cancer)" if probability > 0.5 else "Negative (No Cancer)"
            confidence = probability if probability > 0.5 else 1 - probability
            
            self.result_text.insert(tk.END, f"Prediction: {result}\n")
            self.result_text.insert(tk.END, f"Confidence: {confidence:.2%}\n\n")
            
        else:  # Multi-class classification
            class_idx = np.argmax(prediction[0])
            class_name = self.class_names[class_idx]
            confidence = prediction[0][class_idx]
            
            self.result_text.insert(tk.END, f"Prediction: {class_name}\n")
            self.result_text.insert(tk.END, f"Confidence: {confidence:.2%}\n\n")
            
            # Create sorted list of all class probabilities
            class_probs = [(self.class_names[i], prediction[0][i]) for i in range(len(self.class_names))]
            class_probs.sort(key=lambda x: x[1], reverse=True)
            
            self.result_text.insert(tk.END, "All classes:\n")
            for cls, prob in class_probs:
                self.result_text.insert(tk.END, f"{cls}: {prob:.2%}\n")
        
        # Create and display the probability graph
        self.display_probability_graph(prediction)
    
    def display_probability_graph(self, prediction):
        """Display a graph of prediction probabilities"""
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(5, 4))
        
        # Check if it's binary or multi-class
        if len(prediction[0]) == 1:  # Binary classification
            labels = ["No Cancer", "Cancer"]
            values = [1 - prediction[0][0], prediction[0][0]]
            
            ax.bar(labels, values, color=['green', 'red'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            ax.set_title('Cervial Cancer Prediction Probability')
            
        else:  # Multi-class classification
            values = prediction[0]
            
            # Sort by probability for better visualization
            indices = np.argsort(values)[::-1]
            sorted_classes = [self.class_names[i] for i in indices]
            sorted_values = [values[i] for i in indices]
            
            # Create horizontal bar chart
            bars = ax.barh(sorted_classes, sorted_values, color='skyblue')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.set_title('Class Prediction Probabilities')
            
            # Add value annotations
            for bar, val in zip(bars, sorted_values):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{val:.2f}', va='center')
        
        # Embed the figure in Tkinter
        for widget in self.right_panel.winfo_children():
            if isinstance(widget, FigureCanvasTkAgg):
                widget.get_tk_widget().destroy()
                
        canvas = FigureCanvasTkAgg(fig, master=self.right_panel)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    
    def build_model(self, input_shape, num_classes):
        """Build a CNN model for cancer detection"""
        # Check if we're doing binary or multi-class classification
        final_activation = 'sigmoid' if num_classes == 1 else 'softmax'
        loss_function = 'binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy'
        
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation=final_activation)
        ])
        
        model.compile(
            optimizer='adam',
            loss=loss_function,
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self):
        """Train a new model or retrain the existing model"""
        # Get the data directory
        data_dir = filedialog.askdirectory(title="Select Directory Containing Training Data")
        
        if not data_dir:
            return
            
        try:
            # Check the structure of the data directory
            if self.is_binary_classification(data_dir):
                num_classes = 1
                self.class_names = ["Benign", "Malignant"]  # Default binary names
            else:
                # Get class names from directory structure
                self.class_names = [d for d in os.listdir(data_dir) 
                                  if os.path.isdir(os.path.join(data_dir, d))]
                num_classes = len(self.class_names)
            
            # Data augmentation for training
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=0.2
            )
            
            # Only rescaling for validation
            valid_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )
            
            # Training dataset
            train_generator = train_datagen.flow_from_directory(
                data_dir,
                target_size=self.img_size,
                batch_size=32,
                class_mode='binary' if num_classes == 1 else 'categorical',
                subset='training'
            )
            
            # Validation dataset
            validation_generator = valid_datagen.flow_from_directory(
                data_dir,
                target_size=self.img_size,
                batch_size=32,
                class_mode='binary' if num_classes == 1 else 'categorical',
                subset='validation'
            )
            
            # Update class names if using flow_from_directory
            if num_classes > 1:
                self.class_names = list(train_generator.class_indices.keys())
            
            # Define input shape based on image size and channels
            input_shape = self.img_size + (3,)  # (height, width, channels)
            
            # Build the model
            self.model = self.build_model(input_shape, num_classes)
            
            # Callbacks for model training
            callbacks = [
                ModelCheckpoint('best_cancer_model.h5', save_best_only=True, monitor='val_accuracy'),
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            ]
            
            # Train the model
            history = self.model.fit(
                train_generator,
                epochs=10,  # Adjust as needed
                validation_data=validation_generator,
                callbacks=callbacks
            )
            
            # Save the model
            self.model.save('cancer_detection_model.h5')
            
            # Display training results
            self.display_training_history(history)
            
            # Update status
            self.model_trained = True
            messagebox.showinfo("Success", "Model training completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error training model: {str(e)}")
    
    def is_binary_classification(self, data_dir):
        """Check if the dataset structure indicates binary classification"""
        # Look for binary structure (positive/negative, yes/no, etc.)
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if len(subdirs) == 2:
            binary_pairs = [
                {'positive', 'negative'}, 
                {'yes', 'no'}, 
                {'cancer', 'normal'},
                {'malignant', 'benign'},
                {'1', '0'}
            ]
            
            subdir_set = {d.lower() for d in subdirs}
            for pair in binary_pairs:
                if subdir_set == pair:
                    return True
        
        # If exactly 2 classes but not standard naming, ask user
        if len(subdirs) == 2:
            is_binary = messagebox.askyesno(
                "Dataset Structure", 
                f"Found 2 classes: {subdirs[0]} and {subdirs[1]}. Is this a binary classification problem?"
            )
            return is_binary
            
        return False
    
    def display_training_history(self, history):
        """Display the training history as graphs"""
        # Create a new window for the training history
        history_window = tk.Toplevel(self.root)
        history_window.title("Training History")
        history_window.geometry("800x600")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss')
        ax2.legend()
        
        plt.tight_layout()
        
        # Embed the figure in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=history_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def load_existing_model(self):
        """Load an existing trained model"""
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("H5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if model_path:
            try:
                # Load the model
                self.model = load_model(model_path)
                
                # Try to load class names if available
                class_file = os.path.join(os.path.dirname(model_path), "class_names.txt")
                if os.path.exists(class_file):
                    with open(class_file, 'r') as f:
                        self.class_names = f.read().strip().split('\n')
                else:
                    # Ask user for the number of classes
                    num_classes = self.model.output_shape[1] if len(self.model.output_shape) > 1 else 1
                    
                    if num_classes == 1:
                        self.class_names = ["Benign", "Malignant"]
                    else:
                        # If multi-class, ask user to provide class names
                        class_input = simpledialog.askstring(
                            "Class Names", 
                            f"Enter {num_classes} class names separated by commas:"
                        )
                        if class_input:
                            self.class_names = [c.strip() for c in class_input.split(',')]
                        else:
                            # Default names if user cancels
                            self.class_names = [f"Class {i}" for i in range(num_classes)]
                
                # Update status
                self.model_trained = True
                messagebox.showinfo("Success", "Model loaded successfully!")
                
                # Display model summary in result text
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Model loaded successfully!\n\n")
                self.result_text.insert(tk.END, f"Model file: {os.path.basename(model_path)}\n")
                self.result_text.insert(tk.END, f"Number of classes: {len(self.class_names)}\n")
                self.result_text.insert(tk.END, f"Class names: {', '.join(self.class_names)}\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading model: {str(e)}")

# For simplified imports in Tkinter
from tkinter import simpledialog

if __name__ == "__main__":
    root = tk.Tk()
    app = CancerDetectionApp(root)
    root.mainloop()