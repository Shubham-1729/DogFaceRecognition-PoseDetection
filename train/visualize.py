import cv2
import os

# Paths
images_path = '/home/shubham/Desktop/ogmenProject/DogFaceAndPoseEstimation/dataset/images/train/'
labels_path = '/home/shubham/Desktop/ogmenProject/DogFaceAndPoseEstimation/dataset/labels/train/'

# Define class names and keypoint names
class_names = ['Face']  # Update with your class names
keypoint_names = ['Left Ear', 'Right Ear', 'Left Eye', 'Right Eye', 'Nose']  # Update with your keypoint names

# Iterate through images
for image_file in os.listdir(images_path):
    # Ensure only image files are processed
    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    # Build paths
    image_path = os.path.join(images_path, image_file)
    if(image_file.lower().endswith('.jpg')):
        label_path = os.path.join(labels_path, image_file.replace('.jpg', '.txt'))
    if(image_file.lower().endswith('.png')):
        label_path = os.path.join(labels_path, image_file.replace('.png', '.txt'))
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    height, width = image.shape[:2]

    # Read labels
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        continue

    with open(label_path, 'r') as f:
        for line in f.readlines():
            data = list(map(float, line.strip().split()))
            class_id = int(data[0])
            x_center, y_center, box_width, box_height = data[1:5]
            keypoints = data[5:]

            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add class name near the bounding box
            label_text = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw keypoints and add names
            for i in range(0, len(keypoints), 2):
                kp_x = int(keypoints[i] * width)
                kp_y = int(keypoints[i + 1] * height)
                cv2.circle(image, (kp_x, kp_y), 5, (0, 0, 255), -1)

                # Add keypoint name near the keypoint
                kp_name = keypoint_names[i // 2] if i // 2 < len(keypoint_names) else f"Kp{i // 2}"
                cv2.putText(image, kp_name, (kp_x + 5, kp_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Display image
    cv2.imshow('Image', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

