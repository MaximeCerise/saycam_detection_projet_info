from ultralytics import YOLO
from ut_yolo5 import predict_yolo
from ut_yolo5 import predict_all_frames
import torch
import pandas as pd
from plots_results import plot_focus_map

from extract_frames_from_vidmp4 import extraire_frames

img_path = "vid_frames/vid2_frames/frame_2.jpg"
img2 = "vid_frames/vid2_frames/frame_100.jpg"
model = YOLO('yolo11s.pt')
from ultralytics import YOLO
import pandas as pd

# Define image paths
img_path = "vid_frames/vid2_frames/frame_2.jpg"
img2 = "vid_frames/vid2_frames/frame_100.jpg"

# Load the YOLO model
model = YOLO('yolo11s.pt')

# Perform inference on the second image
output = model(img2)

# Access the detection results
# The output is a list of Results objects (one for each image)
# For a single image, we take the first element of the list
results = output[0]

# Convert results to a pandas DataFrame
# Extract bounding boxes, confidence scores, and class labels
boxes = results.boxes.xyxy.cpu().numpy()  # Bounding boxes in [x1, y1, x2, y2] format
confidences = results.boxes.conf.cpu().numpy()  # Confidence scores
class_ids = results.boxes.cls.cpu().numpy()  # Class IDs
class_names = [results.names[int(cls_id)] for cls_id in class_ids]  # Class names

# Create a DataFrame
results_df = pd.DataFrame({
    'x1': boxes[:, 0],
    'y1': boxes[:, 1],
    'x2': boxes[:, 2],
    'y2': boxes[:, 3],
    'confidence': confidences,
    'class_id': class_ids,
    'name': class_names
})

# Print the results
print(results_df)

# Save the results to a CSV file
results_df.to_csv('detection_results.csv', index=False)

# Visualize the results
results.show()  # This will display the image with bounding boxes
