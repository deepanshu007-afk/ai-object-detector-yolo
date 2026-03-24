import cv2
from ultralytics import YOLO

# 1. Load the pre-trained YOLOv11 nano model (fastest for laptops)
# The model will automatically download the first time you run the script
model = YOLO("yolo11n.pt") 

# 2. Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit the program.")

while True:
    # Capture frame-by-frame
    success, frame = cap.read()
    
    if not success:
        break

    # 3. Run YOLOv11 inference on the frame
    # stream=True is more memory-efficient for real-time video
    results = model.predict(source=frame, stream=True)

    # 4. Process and visualize the results
    for r in results:
        # 'plot()' draws the bounding boxes and labels automatically
        annotated_frame = r.plot()

    # 5. Display the output window
    cv2.imshow("AI Object Detector - 4th Sem Project", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
