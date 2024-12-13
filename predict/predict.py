from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model = YOLO('runs/pose/train15/weights/best.pt')

# Run inference with adjusted confidence threshold
results = model.predict(
    source='/home/shubham/Desktop/ogmenProject/DogFaceAndPoseEstimation/predict/image.png',
    save=True,  # Save the image with bounding boxes and keypoints
    show=True,  # Disable OpenCV display to prevent GUI issues
    conf=0.5    # Lower confidence threshold
)

# Access predictions
predictions = results[0]  # First image

# Extract bounding boxes and keypoints
bbox = predictions.boxes.xywh if predictions.boxes else []
keypoints = predictions.keypoints.xy if predictions.keypoints else []

# Debug: Print keypoints to ensure they are being detected
print("Keypoints detected:", keypoints)

# Predefined keypoint names (adjust these to match your model's output)
keypoint_names =['Left Ear', 'Right Ear', 'Left Eye', 'Right Eye', 'Nose'] 

# Load the image using OpenCV
image = cv2.imread('/home/shubham/Desktop/ogmenProject/DogFaceAndPoseEstimation/predict/image.png')

# Resize the image to match the training size if necessary (640x640)
image_resized = cv2.resize(image, (640, 640))
image_resized_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

# Draw bounding boxes on the image
for box in bbox:
    x_center, y_center, width, height = box
    x1 = int((x_center - width / 2) * image_resized.shape[1])
    y1 = int((y_center - height / 2) * image_resized.shape[0])
    x2 = int((x_center + width / 2) * image_resized.shape[1])
    y2 = int((y_center + height / 2) * image_resized.shape[0])

    # Draw the bounding box
    cv2.rectangle(image_resized_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Check if keypoints exist and draw them
if keypoints is not None and len(keypoints) > 0:
    for i, kp in enumerate(keypoints[0]):  # Only for the first detected object
        x, y = int(kp[0].item()), int(kp[1].item())
        # Debug: Print each keypoint coordinate
        print(f"Keypoint {i} ({keypoint_names[i] if i < len(keypoint_names) else 'Unnamed'}): ({x}, {y})")
        # Draw keypoints as small circles
        cv2.circle(image_resized_rgb, (x, y), 5, (0, 0, 255), -1)
        # Add keypoint name label
        if i < len(keypoint_names):  # Ensure we have a label for the keypoint
            cv2.putText(image_resized_rgb, keypoint_names[i], (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
else:
    print("No keypoints detected.")

# Show the image with bounding boxes and keypoints using Matplotlib
plt.imshow(image_resized_rgb)
plt.axis('off')  # Hide axes
plt.show()

