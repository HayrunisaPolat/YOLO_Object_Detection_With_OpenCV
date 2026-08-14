import cv2
from ultralytics import YOLO
import csv
import time

# Load the YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  

# video file path
video_path = r"C:\YOLO_Object_Detection_With_OpenCV\Videos\crowdedBazaar2.mp4"  

# Initialize video capture
videoCapture = cv2.VideoCapture(video_path) 

displayWidth = 640
displayHeight = 640
frameCount = 0  # Initialize frame count
total_objects_detected = 0  # Initialize total objects detections
total_processing_time_ms = 0  # Initialize total processing time
max_processing_time_ms = 50.0  # Threshold for slow frames in milliseconds

# Open a CSV file to log the results
with open('object_detection_results.csv', mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([' Total Objects Detected ', ' Processing Time (ms) '])  # Write header row

    while videoCapture.isOpened():
        # Read a frame from the video    
        success, frame = videoCapture.read()  

         # Exit the loop if there are no more frames
        if  not success:
            print("Finished processing video or failed to read the frame.")
            break 

        # Increment frame count
        frameCount += 1  
         # Resize the frame to the video dimensions
        resizedFrame = cv2.resize(frame, (displayWidth, displayHeight)) 
        # Start time for processing
        start_time = time.time() 
        # Because verbose is false, the model will not print any output to the console during inference. This is useful for reducing clutter in the console output, especially when processing a large number of frames.
        results = yolo_model(resizedFrame, verbose = False)  # Perform object detection on the frame

        # Loop through the results and extract bounding boxes, confidence scores, and class IDs
        for result in results:
             # Get bounding box coordinates, confidence scores, and class IDs from the result
            boxes = result.boxes.xyxy.cpu().numpy() 
            # Get confidence scores 
            confidences = result.boxes.conf.cpu().numpy()  
             # Get class IDs
            class_ids = result.boxes.cls.cpu().numpy() 
            # Count the number of objects detected in the current frame
            total_objects_detected += len(boxes)

            # Loop through each detected object and draw bounding boxes and labels
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, box)  
                # Convert class ID to integer
                class_id = int(class_id)  
                # Get class name from class ID
                name = result.names[class_id]  
                # Convert confidence score to float
                conf = float(conf)
                # Create text with class name and confidence  
                objectText = f"{name}: {conf:.2f}" 

                # Draw bounding green box
                cv2.rectangle(resizedFrame, (x1, y1), (x2, y2), (0, 255, 0), 2)  
                # Put label text
                cv2.putText(resizedFrame, objectText, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  

        # End time for processing
        end_time = time.time() 
        # Calculate processing time in milliseconds
        processing_time_ms = (end_time - start_time) * 1000  
        # Accumulate total processing time
        total_processing_time_ms += processing_time_ms 

        # Check if the processing time exceeds the threshold
        if processing_time_ms > max_processing_time_ms:  
            print("UYARI: Bu hiz Jetson'da calismaz, optimize et!")

        # Write to CSV every 100 frames
        if( frameCount % 100 == 0 ):  
            avg_processing_time = total_processing_time_ms / 100
            #  Total objects detected in the last 100 frames and average processing time
            csv_writer.writerow([total_objects_detected, f"{avg_processing_time:.2f}"]) 
            # For the next 100 frames, reset the counters
            total_processing_time_ms = 0.0
            total_objects_detected = 0

        # Display the frame with detections 
        cv2.imshow("YOLOv8 Object Detection", resizedFrame) 
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

# Release video capture and close all OpenCV windows
videoCapture.release()
cv2.destroyAllWindows()