#Project Name - GAN MODEL
#Author - Rhythm
#Course - COMP 3710
#TRUID - T00684614
#Thompson Rivers University

#Importing all vital librarues
import cv2
import os

# Define input and output directories
input_dir = r'F:\GAN_Project\data\training_data'
output_dir = r'F:\GAN_Project\data\resized_paintings'

# Define target size for resizing
target_size = (224, 224)

# Loop through each file in the input directory
for file_name in os.listdir(input_dir):
    # Load image from file
    input_path = os.path.join(input_dir, file_name)
    img = cv2.imread(input_path)

    # Resize image to target size
    resized_img = cv2.resize(img, target_size)

    # Save resized image to output directory
    output_path = os.path.join(output_dir, file_name)
    cv2.imwrite(output_path, resized_img)
