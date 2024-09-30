# test.py
import os
import cv2
import numpy as np
from tensorflow import keras

# Step 1: Load the trained model
model_path = './ai_face_detector_model.keras'  # Path to your saved model
model = keras.models.load_model(model_path)
print("Model loaded successfully.")

# Step 2: Load and preprocess the test image
def load_and_preprocess_test_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:  # Check if the image was loaded successfully
        image = cv2.resize(image, (128, 128))  # Resize to match model input
        image = np.array(image) / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    else:
        print("Error: Image not found.")
        return None

# Step 3: Test the model on a specific image
def test_image(image_path):
    test_image = load_and_preprocess_test_image(image_path)
    
    if test_image is not None:
        prediction = model.predict(test_image)
        predicted_label = 1 if prediction[0][0] > 0.5 else 0  # Threshold of 0.5 for binary classification
        result = "AI-generated" if predicted_label == 1 else "Real"
        print(f"The image '{os.path.basename(image_path)}' is predicted to be: {result}")

# Main execution
if __name__ == "__main__":
    # test_image_path = './test_real.jpg'  
    test_image_path = './test_fake.png'  
    test_image(test_image_path)
