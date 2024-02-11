import os
import cv2
import imutils

from src.cnsts import DATASET_DIR

IMG_NAME: str = 'london_driving.png'
IMG_PATH: str = os.path.join(DATASET_DIR, IMG_NAME)

# Load an image
img = cv2.imread(IMG_PATH)

# Get image size
img_height, img_width, img_channels = img.shape
print(f'Image Height: {img_height}, Image Width: {img_width}, Image Channels: {img_channels}')

# Display the image
cv2.imshow('London Driving', img)
cv2.waitKey(0)

# Access pixel values
pixel = img[100, 150]  # First row, second column. Remember that OpenCV uses BGR instead of RGB for pixel values
print(f'Blue Intensity: {pixel[0]}, Green Intensity: {pixel[1]}, Red Intensity: {pixel[2]}')

# Extract a region of interest
initial_height, initial_width = 100, 100
final_height, final_width = 200, 200
roi = img[initial_height:final_height, initial_width:final_width]  # First heights, then widths due to numpy array standard

# Display the region of interest
cv2.imshow('Region of Interest', roi)
cv2.waitKey(0)

# Save the region of interest
ROI_NAME: str = 'london_driving_roi.png'
ROI_PATH: str = os.path.join(DATASET_DIR, ROI_NAME)
cv2.imwrite(ROI_PATH, roi)

# Resize the image
resized_img = cv2.resize(img, (300, 300))
cv2.imshow('Resized London Driving without keeping aspect ratio', resized_img)
cv2.waitKey(0)

# Resize the image keeping aspect ratio
resized_img_with_same_aspect_ratio = imutils.resize(img, width=300)
cv2.imshow('Resized London Driving keeping aspect ratio', resized_img_with_same_aspect_ratio)
cv2.waitKey(0)

# Rotate the image
rotated_img = imutils.rotate(img, 45)
cv2.imshow('Rotated London Driving', rotated_img)
cv2.waitKey(0)

# Rotate the image without cropping
rotated_img_without_cropping = imutils.rotate_bound(img, 45)
cv2.imshow('Rotated London Driving without cropping', rotated_img_without_cropping)
cv2.waitKey(0)

# Draw a rectangle
img_copy_for_draw = img.copy()
start_point = (100, 100)
end_point = (200, 200)
cv2.rectangle(img_copy_for_draw, start_point, end_point, (0, 255, 0), 2)

# Draw a circle
center_coordinates = (150, 150)
radius = 50
cv2.circle(img_copy_for_draw, center_coordinates, radius, (0, 0, 255), 2)

# Draw a line
start_point = (100, 100)
end_point = (200, 200)
cv2.line(img_copy_for_draw, start_point, end_point, (255, 0, 0), 2)
cv2.imshow('London Driving with rectangle, circle and line', img_copy_for_draw)
cv2.waitKey(0)

# Put text on the image
font = cv2.FONT_HERSHEY_SIMPLEX
start_point = (50, 50)
font_scale = 1
font_color = (255, 255, 255)
line_type = 2
cv2.putText(img_copy_for_draw, 'London Driving', start_point, font, font_scale, font_color, line_type)
cv2.imshow('London Driving with text', img_copy_for_draw)
cv2.waitKey(0)

