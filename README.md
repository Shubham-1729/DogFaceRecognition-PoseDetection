# Dog Face Recognition and Pose Detection

This project involves training a model for detecting dog faces and their keypoints, such as left ear, right ear, left eye, right eye, and nose. The model uses YOLO for object detection and pose estimation. This README provides information on how to set up, train, and use the project.

## Features

- Detects dog faces and keypoints.
- Keypoints include:
  - Left Ear
  - Right Ear
  - Left Eye
  - Right Eye
  - Nose
- Outputs annotated images with bounding boxes and keypoints.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8+
- PyTorch
- OpenCV
- YOLOv8
- Git

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Shubham-1729/DogFaceRecognition-PoseDetection.git
   cd DogFaceRecognition-PoseDetection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate    # For Windows
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Add `venv` to `.gitignore` (optional):
   ```bash
   echo "venv/" >> .gitignore
   ```

## Dataset Preparation

1. Collect and annotate images of dogs with bounding boxes and keypoints using a tool like [LabelImg](https://github.com/tzutalin/labelImg).

2. Organize the dataset as follows:
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   ┌── labels/
   ├── val/
       ├── images/
       └── labels/
   ```

3. Update the dataset YAML file (e.g., `dog_dataset.yaml`):
   ```yaml
   train: dataset/train/images
   val: dataset/val/images
   nc: 1
   names: ['Dog Face']
   ```

## Training the Model

1. Train the model using YOLOv8:
   ```bash
   yolo pose train data=dog_dataset.yaml model=yolov8n-pose.pt epochs=100 imgsz=640
   ```

2. After training, the best weights will be saved in the `runs/pose/train/weights/best.pt` directory.

## Running Inference

1. Use the trained model to detect faces and keypoints:
   ```python
   from ultralytics import YOLO
   import cv2
   import matplotlib.pyplot as plt

   # Load the trained model
   model = YOLO('runs/pose/train/weights/best.pt')

   # Run inference
   results = model.predict(source='path/to/image.jpg', conf=0.5, save=True)

   # Display results
   results.show()
   ```

2. Annotated images will be saved in the `runs/pose/predict` directory.

## Troubleshooting

- **Keypoints Not Detected:** Ensure that the dataset has accurate annotations and enough samples for training.
- **Large File Issues on GitHub:** Use Git LFS for managing large files:
  ```bash
  git lfs install
  git lfs track '*.pt'
  ```

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [LabelImg](https://github.com/tzutalin/labelImg)

Feel free to contribute to this project by opening issues or submitting pull requests!

