# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 21:25:17 2024

@author: Lenovo
"""

import cv2
import numpy as np
import tkinter as tk

# Load YOLO
yolo_config_path = 'yolov3.cfg'  # Path to the config file
yolo_weights_path = 'yolov3.weights'  # Path to the YOLO weights file
coco_names_path = 'coco.names'  # Path to the COCO labels file

# Load the YOLO model
net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)

# Load the COCO names (80 classes in total)
with open(coco_names_path, 'r') as f:
    classes = f.read().strip().split("\n")

# Get the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define a crowd threshold
CROWD_THRESHOLD = 30  # Set the threshold limit for people
SPACE_THRESHOLD = 0.5  # Space threshold for the crowd density analysis

# Define function to detect, count people, and analyze space
def detect_and_analyze_space(frame):
    height, width, _ = frame.shape
    total_frame_area = width * height  # Total area of the frame

    # Create a blob from the frame (scaling, normalization, resizing)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    # Get YOLO network predictions
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    total_person_area = 0  # Initialize total area occupied by people

    # Loop through each of the detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Only consider detections for the "person" class (class_id = 0)
            if class_id == 0 and confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the coordinates of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # Calculate the area occupied by this person
                person_area = w * h
                total_person_area += person_area

    # Apply Non-Maximum Suppression (NMS) to reduce multiple overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Initialize count
    person_count = 0

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]

            # Draw bounding box and label for each person detected
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Count the number of people
            person_count += 1

    # Display the count on the frame
    cv2.putText(frame, f"People Count: {person_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Calculate crowd-to-space ratio
    crowd_to_space_ratio = total_person_area / total_frame_area

    # Display the crowd-to-space ratio on the frame
    cv2.putText(frame, f"Space Occupied: {crowd_to_space_ratio:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    # Check if the crowd threshold is exceeded
    if person_count > CROWD_THRESHOLD:
        trigger_message = "ALERT! Crowd size exceeds threshold!"
        print(trigger_message)
        cv2.putText(frame, trigger_message, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Check if the space occupancy is above the space threshold
    if crowd_to_space_ratio > SPACE_THRESHOLD:
        space_alert_message = "ALERT! Crowd is occupying too much space!"
        print(space_alert_message)
        cv2.putText(frame, space_alert_message, (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    return frame, person_count, crowd_to_space_ratio

# Open video file or webcam feed
#video_path = "C:\\Users\\Lenovo\\Desktop\\imp\\minorProject\\1338598-hd_1920_1080_30fps.mp4"  # Replace with your video file or use 0 for live webcam
video_path = "C:\\Users\\Lenovo\\Desktop\\imp\\minorProject\\1338598-hd_1920_1080_30fps.mp4"
cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(0)
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.withdraw()  # Close the tkinter window

# Calculate window size to be half of the screen size
window_width = screen_width // 2
window_height = screen_height // 2

# Calculate the top-left corner of the window to center it
window_x = (screen_width - window_width) // 2
window_y = (screen_height - window_height) // 2

# Create a named window with normal window size
cv2.namedWindow('People Detection, Counting, and Space Analysis', cv2.WINDOW_NORMAL)

# Set the window size and position
cv2.resizeWindow('People Detection, Counting, and Space Analysis', window_width, window_height)
cv2.moveWindow('People Detection, Counting, and Space Analysis', window_x, window_y)


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Detect and analyze space in the frame
    frame, count, ratio = detect_and_analyze_space(frame)

    # Show the output frame
    cv2.imshow('People Detection, Counting, and Space Analysis', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
