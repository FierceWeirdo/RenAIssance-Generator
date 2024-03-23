#Project Name - GAN MODEL
#Author - Rhythm
#Course - COMP 3710
#TRUID - T00684614
#Thompson Rivers University

#Importing all vital librarues
import cv2
import numpy as np

# Load the two images
img1 = cv2.imread('input_image4.jpg')
img2 = cv2.imread('output_image4.jpg')

# Define the skin tone range in LAB color space
lower_skin = np.array([0, 133, 77], dtype=np.uint8)
upper_skin = np.array([255, 173, 127], dtype=np.uint8)
#Any pixel that falls in this range will be considered a skin pixel

# Convert the images from BGR to LAB color space
img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

# Extract the skin tone pixels from each image
mask1 = cv2.inRange(img1_lab, lower_skin, upper_skin)
mask2 = cv2.inRange(img2_lab, lower_skin, upper_skin)

# Calculate the percentage of skin tone pixels in each image
total_pixels1 = np.prod(img1.shape[:2])
skin_pixels1 = np.count_nonzero(mask1)
skin_percent1 = skin_pixels1 / total_pixels1

total_pixels2 = np.prod(img2.shape[:2])
skin_pixels2 = np.count_nonzero(mask2)
skin_percent2 = skin_pixels2 / total_pixels2

# Compare the skin tone percentages to see if they match
threshold = 0.09 
print("FOR IMAGE 4: ")
if abs(skin_percent1 - skin_percent2) <= threshold:
    print('The skin tones match!')
else:
    print('The skin tones do not match.')
