import cv2

# Read the camera
camera = cv2.VideoCapture(0)  # 0 is the default camera, use whatever index your camera is.
# To read a video file, use cv2.VideoCapture('path_to_video_file.mp4')

# Loop through the frames
while True:
    # Read the frame
    ret, frame = camera.read()
    if frame is None:  # If the frame is empty, break the loop
        break

    # Show the frame
    cv2.imshow('Video', frame)

    key = cv2.waitKey(1)  # Wait for 1 millisecond
    if key == ord('q'):  # If the user presses q, break the loop
        break

# Release the camera
camera.release()
# Destroy all windows
cv2.destroyAllWindows()
