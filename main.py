from ultralytics import YOLO
import cv2

# Load model
license_plate_detector = YOLO("./models/license_plate_detector.pt")

# Load video
cap = cv2.VideoCapture("./data/input/traffic.mp4")

# Get frame width, height, and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object to save the video
output_path = "./data/output/annotated_traffic.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Define codec (for .mp4)
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Read frames and process
ret = True
while ret:
    ret, frame = cap.read()
    
    if ret:    
        # Detect and track license plates
        license_plates = license_plate_detector.track(frame, persist=True)

        # Annotate frame with detected license plates
        for plate in license_plates:
            for bbox in plate.boxes:
                x1, y1, x2, y2 = bbox.xyxy[0]  # Get bounding box coordinates
                
                # Convert to integers
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

        # Display the annotated frame
        cv2.imshow("YOLO Tracking", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
out.release()  # Release the VideoWriter object
cap.release()
cv2.destroyAllWindows()
