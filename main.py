import os
from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf

# Set the environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Specify the input and output directories
input_directory = "C:/Users/Pubudu Madusith/Pictures/Camera Roll"  # Change this to your image directory
# output_directory = "D:/3yp_Project/ESP32/PythonTensorFlow/Processed"

# Create the output directory if it doesn't exist
# os.makedirs(output_directory, exist_ok=True)

# List all files in the input directory
image_files = [f for f in os.listdir(input_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(input_directory, image_file)

    # Read the image from the file
    image = cv2.imread(image_path)

    # Resize the image
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Image from Directory", image)
    
    # Make the image a numpy array and reshape it to the model's input shape
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predict the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print(f"Image: {image_file}, Class: {class_name[2:]}, Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%")
    print(class_name[2:])
    
    # os.remove(image_path)
    # print(f"Original image {image_file} deleted.")

    # Wait for a key press to continue processing the next image
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

# Destroy the window after processing all images
cv2.destroyAllWindows()

