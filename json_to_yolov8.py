import json
import os
import shutil

json_dir = "/home/hawkjack/Downloads/2/"
label_dir = "/home/hawkjack/Downloads/2/label/"
source_img = "/home/hawkjack/Downloads/beagle/"
dest_img = "/home/hawkjack/Downloads/2/img/"


def json_to_yolov8(
    json_dir, output_dir, class_id=0, image_src_dir=None, image_dst_dir=None
):
    """
    Convert JSON annotations to YOLOv8 format and optionally copy images.

    Parameters:
    - json_dir: Directory containing JSON files.
    - output_dir: Directory to save YOLOv8 annotation files.
    - class_id: Class ID for objects (default is 0).
    - image_src_dir: Directory containing source images (optional).
    - image_dst_dir: Directory to copy images to (optional).
    """
    os.makedirs(output_dir, exist_ok=True)
    if image_dst_dir:
        os.makedirs(image_dst_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, "r") as file:
            data = json.load(file)

        # Image metadata
        image_name = data["item"]["name"]
        image_width = data["item"]["slots"][0]["width"]
        image_height = data["item"]["slots"][0]["height"]

        # Parse annotations
        yolov8_annotations = []
        bounding_box = None
        keypoints = {}

        for annotation in data["annotations"]:
            if "bounding_box" in annotation:
                bbox = annotation["bounding_box"]
                x_center = (bbox["x"] + bbox["w"] / 2) / image_width
                y_center = (bbox["y"] + bbox["h"] / 2) / image_height
                width = bbox["w"] / image_width
                height = bbox["h"] / image_height
                bounding_box = [x_center, y_center, width, height]
            elif "keypoint" in annotation:
                kp_name = annotation["name"]
                kp_x = annotation["keypoint"]["x"] / image_width
                kp_y = annotation["keypoint"]["y"] / image_height
                keypoints[kp_name] = (kp_x, kp_y)

        # Ensure all required keypoints are present
        keypoint_order = ["Left-Ear", "Right-Ear", "Left-Eye", "Right-Eye", "Nose"]
        keypoint_coords = []
        for kp in keypoint_order:
            if kp in keypoints:
                keypoint_coords.extend(keypoints[kp])
            else:
                keypoint_coords.extend([0, 0])  # Placeholder if keypoint is missing

        # Combine all into a YOLOv8 annotation line
        if bounding_box:
            annotation_line = f"{class_id} " + " ".join(
                map(str, bounding_box + keypoint_coords)
            )
            yolov8_annotations.append(annotation_line)

        # Write YOLOv8 annotation file
        output_file = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".txt")
        with open(output_file, "w") as txt_file:
            txt_file.write("\n".join(yolov8_annotations))

        # Copy the image if paths are provided
        if image_src_dir and image_dst_dir:
            src_image_path = os.path.join(image_src_dir, image_name)
            dst_image_path = os.path.join(image_dst_dir, image_name)
            if os.path.exists(src_image_path):
                try:
                    shutil.copy(src_image_path, dst_image_path)
                    print(f"Copied: {src_image_path} -> {dst_image_path}")
                except Exception as e:
                    print(f"Error copying {src_image_path}: {e}")
            else:
                print(f"Image not found: {src_image_path}")


if __name__ == "__main__":
    # Input and output directories
    json_dir = input("Enter the path to the directory containing JSON files: ").strip()
    output_dir = input("Enter the path to save YOLOv8 annotation files: ").strip()

    # Image directories (optional)
    copy_images = (
        input("Do you want to copy images? (yes/no): ").strip().lower() == "yes"
    )
    if copy_images:
        image_src_dir = input("Enter the path to the source image directory: ").strip()
        image_dst_dir = input(
            "Enter the path to the destination image directory: "
        ).strip()
    else:
        image_src_dir = None
        image_dst_dir = None

    # Convert JSON to YOLOv8 and copy images
    print("Converting JSON annotations to YOLOv8 format...")
    json_to_yolov8(
        json_dir, output_dir, image_src_dir=image_src_dir, image_dst_dir=image_dst_dir
    )
    print(f"Conversion complete! YOLOv8 annotations saved in '{output_dir}'.")
    if copy_images:
        print(f"Images copied to '{image_dst_dir}'.")
