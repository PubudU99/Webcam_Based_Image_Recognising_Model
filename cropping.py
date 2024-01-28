from PIL import Image
import os

# Input directory path
input_directory = "C:/Users/Pubudu Madusith/Pictures/Camera Roll"

# Output directory path for cropped images
output_directory = "C:/Users/Pubudu Madusith/Pictures/Camera Roll/Cropped"
os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist

# List all image files in the input directory
image_files = [f for f in os.listdir(input_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# Loop through each image file and crop
for image_file in image_files:
    # Construct the full path for the image file
    image_path = os.path.join(input_directory, image_file)

    # Opens an image in RGB mode
    im = Image.open(image_path)

    # Size of the image in pixels (size of the original image)
    width, height = im.size

    # Setting the points for cropped image
    left_crop = 100  # Adjust the left crop value as needed
    right_crop = width - 100  # Adjust the right crop value as needed
    top = 0
    bottom = height

    # Cropped image of the specified dimensions
    im_cropped = im.crop((left_crop, top, right_crop, bottom))

    # Construct the full path for the cropped image
    output_path = os.path.join(output_directory, f"Cropped_{image_file}")

    # Save the cropped image to the output directory
    im_cropped.save(output_path)

    # Optionally, show the cropped image
    im_cropped.show()

print("Cropping and saving complete.")
