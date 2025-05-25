"""Video capture implementation"""
import cv2
import os
import time
import torch
from cv2.typing import MatLike

class VideoCapture:
    """Class implementation of video capture."""

    def __init__(self, storage_path: str, camera_index: int = 0) -> None:
        """Create new video capture implementation.
        
            @param storage_path Path used to store images.
            @param camera_index Index of the camera to use (default = internal camera).
        """
        if storage_path is None or storage_path == "":
            raise ValueError("Invalid storage path!")
    
        # Create video capture with OpenCV.
        self._capture = cv2.VideoCapture(camera_index)

        # Create Yolov5 model.
        self._model = torch.hub.load('ultralytics/yolov5', 'yolov5m') 
        
        # Store path for saving images.
        self._storage_path = storage_path

        # Stored results.
        self._results = None

        if not self._capture.isOpened():
            raise ValueError("Failed to open the camera!")

    def __del__(self) -> None:
        """Delete camera implementation."""
        self._capture.release()

    @staticmethod
    def close() -> bool:
        """ Close video capture if 'q' is pressed."""
        return cv2.waitKey(1) & 0xFF == ord('q')

    def run(self, accuracy: float = 0.50) -> None:
        assert accuracy >= 0.0 and accuracy <= 1.0, "Invalid accuracy"

        # Read video capture.
        ret, frame = self._capture.read()
        if not ret:
            raise ValueError("Failed to read from video capture stream!")

        # Perform object recognition with YOLOv5.
        self._results = self._model(frame)

        # Filter and save data.
        self._filter(accuracy, frame)

        # Add bounding boxes for the image.
        self._results.render()  

        # Show the result.
        cv2.imshow("YOLOv5 Object Detection", self._results.ims[0])

    def _filter(self, accuracy: float, frame: MatLike) -> None:
        # Filter detections with given accuracy.
        for i, det in enumerate(self._results.pred[0]):
            score = det[4].item()
            if score >= accuracy:
                # Convert YOLO format (center_x, center_y, width, height) to (xmin, ymin, xmax, ymax)
                x_center, y_center, width, height = det[:4]
                xmin = int(x_center - width / 2)
                ymin = int(y_center - height / 2)
                xmax = int(x_center + width / 2)
                ymax = int(y_center + height / 2)

                # Ensure coordinates are within image bounds
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(frame.shape[1], xmax), min(frame.shape[0], ymax)

                # Crop the detected object correctly
                detected_object = frame[ymin:ymax, xmin:xmax].copy()

                # Save the object image
                obj_name = self._results.names[int(det[5].item())]
                obj_dir = os.path.join(self._actual_path(), obj_name)
                os.makedirs(obj_dir, exist_ok=True)
                timestamp = int(time.time() * 1000)
                filename = os.path.join(obj_dir, f'{obj_name}_{timestamp}.jpg')

                cv2.imwrite(filename, detected_object)

    def _actual_path(self) -> str:
        """Return the path to store data."""
        return self._storage_path if self._storage_path else "detected_objects"
    