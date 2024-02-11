import os
import cv2
import imutils

from src.cnsts import DATASET_DIR

IMG_NAME: str = 'tetris_blocks_2.png'

# Load an image
img = cv2.imread(os.path.join(DATASET_DIR, IMG_NAME))

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edged_img = cv2.Canny(gray_img, 30, 150)  # minVal and maxVal are the minimum and maximum intensity gradients

# Display the original image, grayscale image, and edge-detected image
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.imshow('Grayscale Image', gray_img)
cv2.waitKey(0)
cv2.imshow('Edge-detected Image', edged_img)
cv2.waitKey(0)

# Find contours
contours = cv2.findContours(edged_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # CHAIN_APPROX_SIMPLE stores only the endpoints of the contours
contours = imutils.grab_contours(contours)

# Draw the contours
output_img = img.copy()
cv2.drawContours(output_img, contours, -1, (0, 255, 0), 2)  # -1 means draw all the contours, (0, 255, 0) is the color, 2 is the thickness
cv2.putText(output_img, f'Found {len(contours)} objects', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow('Contours', output_img)
cv2.waitKey(0)

# Loop through the contours
for i, contour in enumerate(contours):
    # Compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(output_img, f'Object {i + 1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.imshow('Bounding Box', output_img)
cv2.waitKey(0)

