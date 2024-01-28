import os
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import time

# Set the environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Specify the output directory
output_directory = "C:/Users/Pubudu Madusith/Pictures/Camera Roll/Cropped"
os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist

# Specify the index of the external webcam (you may need to adjust this)
external_webcam_index = 1  # Change this to the correct index for your external webcam

# Open the external webcam
cap = cv2.VideoCapture(external_webcam_index)

while True:
    # Capture frame from the external webcam
    ret, frame = cap.read()
    
    # Display the captured frame
    cv2.imshow("External Webcam", frame)

    # Convert the frame to a PIL image
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform cropping
    width, height = im.size
    left_crop = 100  # Adjust the left crop value as needed
    right_crop = width - 100  # Adjust the right crop value as needed
    top = 0
    bottom = height
    im_cropped = im.crop((left_crop, top, right_crop, bottom))

    # Resize the cropped image to match the model's input shape
    im_cropped = im_cropped.resize((224, 224))

    # Construct the full path for the cropped image
    output_path = os.path.join(output_directory, f"Cropped_frame.jpg")

    # Save the cropped image to the output directory
    im_cropped.save(output_path)

    # Convert the cropped image to a numpy array
    cropped_image = np.asarray(im_cropped, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    cropped_image = (cropped_image / 127.5) - 1

    # Predict the model
    prediction = model.predict(cropped_image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print(f"Class: {class_name[2:]}, Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%")

    # Wait for 2 seconds before processing the next frame
    time.sleep(2)

    # Check for the escape key (27 is the ASCII for the esc key)
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break

# Release the external webcam and destroy the window
cap.release()
cv2.destroyAllWindows()
