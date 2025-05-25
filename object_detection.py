"""Video capture test"""
from video_capture import VideoCapture

STORAGE_PATH : str = "C:/yrgo/img_opencv"

def main() -> None:
    # Create video capture.
    capture = VideoCapture(STORAGE_PATH)

    # Run video capture until the q button is pressed.
    while not VideoCapture.close():
        capture.run()

if __name__ == "__main__":
    # Invoke the main function is this is the startup script.
    main()
