import os
import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

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

# Step 3: Generate Grad-CAM heatmap
def generate_grad_cam_heatmap(image_path, class_index):
    # Load and preprocess the image
    original_image = cv2.imread(image_path)
    processed_image = load_and_preprocess_test_image(image_path)

    # Get the last convolutional layer
    layer_name = model.layers[-3].name  # Adjust this index if necessary

    # Create a new model that outputs the last conv layer and the output
    grad_model = keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # Run the model on the input image to ensure it's been called
    with K.GradientTape() as tape:
        conv_outputs, predictions = grad_model(processed_image)  # Call the model
        loss = predictions[0][class_index]

    # Compute the gradients of the output with respect to the last convolutional layer
    grads = tape.gradient(loss, conv_outputs)

    # Compute the mean of the gradients
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # Get the output of the last convolutional layer
    conv_outputs = conv_outputs[0]

    # Multiply the pooled gradients by the output of the last convolutional layer
    heatmap = conv_outputs @ pooled_grads.numpy()
    heatmap = K.relu(heatmap)

    # Normalize the heatmap
    heatmap /= np.max(heatmap)

    # Resize the heatmap to the original image size
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # Convert to 8-bit values
    return heatmap



# Step 4: Test the model on a specific image and generate heatmap
def test_image(image_path):
    test_image = load_and_preprocess_test_image(image_path)
    
    if test_image is not None:
        prediction = model.predict(test_image)
        predicted_label = 1 if prediction[0][0] > 0.5 else 0  # Threshold of 0.5 for binary classification
        result = "AI-generated" if predicted_label == 1 else "Real"
        print(f"The image '{os.path.basename(image_path)}' is predicted to be: {result}")

        # Generate and display heatmap if the image is predicted to be AI-generated
        if predicted_label == 1: 
            heatmap = generate_grad_cam_heatmap(image_path, class_index=1)  # Use index 1 for AI-generated
            # Display the heatmap
            plt.imshow(heatmap, cmap='jet', alpha=0.5)
            plt.axis('off')
            plt.title("Grad-CAM Heatmap")
            plt.show()

# Main execution
if __name__ == "__main__":
    test_image_path = './test_fake.png'  
    test_image(test_image_path)
