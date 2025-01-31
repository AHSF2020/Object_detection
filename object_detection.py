import cv2
from cv2.typing import MatLike
import torch
import os
import time

class VideoCapture:
    """ @brief Class implememtation if video capture."""

    def __init__(self, camera_index: int = 0, path: str = "img_opencv") -> None:
        """ @brief Create new video capture implementation.
        
            @param camera_index Index of the camera to use (default = internal camera).
        """
        # Create video capture with OpenCV.
        self._cap = cv2.VideoCapture(camera_index)

        # Create Yolov5 model.
        self._model = torch.hub.load('ultralytics/yolov5', 'yolov5m') 
        
        # Store path for saving images.
        self._path = path

        # Stored results.
        self._results = None

        if not self._cap.isOpened():
            raise ValueError("Failed to open the camera!")

    def __del__(self) -> None:
        """ @brief Delete camera implementation."""
        self._cap.release()

    def run(self, accuracy: float = 0.50) -> None:
        assert accuracy >= 0.0 and accuracy <= 1.0, "Invalid accuracy"

        # Read video capture.
        ret, frame = self._cap.read()
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
        """@brief Return the path to store data"""
        return self._path if self._path else "detected_objects"

def qExit() -> bool:
    # Close if 'q' is pressed.
    return cv2.waitKey(1) & 0xFF == ord('q')

def destroyAllWindows() -> None:
    cv2.destroyAllWindows()

def main() -> None:
    cap1 = VideoCapture(0, "C:/Users/Admin/Desktop/Visual Studio/img_opencv")
    while not qExit():
        cap1.run()

if __name__ == "__main__":
    # Invoke the main function is this is the startup script.
    main()
