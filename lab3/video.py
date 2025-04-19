import cv2 as cv
import time

# Open the camera (0 usually means the default webcam)
camera = cv.VideoCapture(0)

# Get the camera's width and height
width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))

# Let the camera warm up
time.sleep(2)

while True:
    ret, frame = camera.read()  # Read a frame from the camera
    if not ret:
        print("Error: Failed to capture image.")
        break  # Exit the loop if the frame capture failed

    # Check if the frame has valid width and height
    if frame.shape[0] > 0 and frame.shape[1] > 0:
        cv.imshow("Camera Feed", frame)  # Show the captured frame
        
    else:
        print("Error: Invalid frame dimensions.")

    # Wait for a key press; if 'q' is pressed, exit the loop
    if cv.waitKey(1) & 0xFF == ord('q'):  # 1 ms delay for each frame
        break
    #wait for 1 ms if no key pressed move to second frame

# Release the camera and close all OpenCV windows
camera.release()#stop conncetion to camera
cv.destroyAllWindows()#close all windows
