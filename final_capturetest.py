# -*- coding: utf-8 -*- 

from picamera2 import Picamera2
import cv2
import time
import datetime
import os

# Create a Picamera2 instance
picam2 = Picamera2()

# Set the desired resolution
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

# Create a preview configuration (for real-time processing)
camera_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": IMAGE_SIZE}
)
camera_config["controls"]["FrameDurationLimits"] = (33333, 33333)  # Set to 30fps

# Apply the camera settings
picam2.configure(camera_config)

# Start the camera
picam2.start()

# Real-time capture loop
i = 0
capture_start_time = time.time()  # Record the start time
frames_captured = 0  # Number of frames captured

# Create the folder at the start of the program
now = datetime.datetime.now()
folder_name = now.strftime('%Y-%m-%d_%H-%M')
directory = f'./test_video/{folder_name}/'

# Create the folder if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Set the capture duration to 3 minutes (180 seconds)
end_time = capture_start_time + 120  # Current time + 180 seconds (3 minutes)

try:
    while True:
        # Check the current time
        current_time = time.time()

        # Stop the capture if 3 minutes have passed
        if current_time >= end_time:
            print("3 minutes have passed, stopping the capture.")
            break

        # Capture an image
        rgb_image = picam2.capture_array()

        # Update the number of frames captured
        frames_captured += 1

        # Calculate the number of frames captured per second
        elapsed_time = current_time - capture_start_time

        # Print the number of frames captured per second every 1 second
        if elapsed_time >= 1.0:
            print(f"Images captured per second: {frames_captured}")
            capture_start_time = current_time  # Reset the time
            frames_captured = 0  # Reset the frame count

        # Set the image file name
        filename = f"{i}.jpg"
        
        # Save the image
        #cv2.imwrite(directory + filename, rgb_image)

        i += 1  # Increment the counter

        # Show the live preview (resize to 200x150)
 
        cv2.imshow("Live Preview", rgb_image)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

# Release resources
cv2.destroyAllWindows()
picam2.close()
