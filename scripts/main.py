# main.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

# Custom progress callback
class ProgressCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        total_epochs = self.params['epochs']
        percentage = (epoch + 1) / total_epochs * 100
        print(f"\nEpoch {epoch + 1}/{total_epochs} - {percentage:.2f}% complete")

# Step 1: Load and preprocess images
def load_and_preprocess_images(directory, label):
    images = []
    labels = []
    
    for image_name in os.listdir(directory):
        image_path = os.path.join(directory, image_name)
        image = cv2.imread(image_path)
        
        if image is not None:  # Check if the image was loaded successfully
            image = cv2.resize(image, (128, 128))  # Resize to 128x128
            images.append(image)
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    images = images / 255.0  # Normalize images
    return images, labels

# Load datasets
real_faces_dir = r"C:\Users\hurem\OneDrive\Desktop\ai_v2\data\real_faces"
ai_faces_dir = r"C:\Users\hurem\OneDrive\Desktop\ai_v2\data\ai_faces"

real_images, real_labels = load_and_preprocess_images(real_faces_dir, label=0)  # Label 0 for real faces
ai_images, ai_labels = load_and_preprocess_images(ai_faces_dir, label=1)  # Label 1 for AI-generated faces

# Combine datasets
images = np.concatenate((real_images, ai_images))
labels = np.concatenate((real_labels, ai_labels))

# Step 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 3: Build the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Load model if exists
model_path = './ai_face_detector_model.h5'
weights_path = './ai_face_detector_model.weights.h5' 

if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded existing model weights.")

# Create a callback to save the model's weights
checkpoint = ModelCheckpoint(weights_path, save_weights_only=True, save_best_only=True, verbose=1)

# Step 5: Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, 
          callbacks=[ProgressCallback(), checkpoint])

# Step 6: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

# Optional: Save the complete model
# model.save(model_path)


# TODO
model.save('./ai_face_detector_model.keras')
