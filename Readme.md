# Human Activity Detection using MediaPipe and OpenCV

This project utilizes the MediaPipe and OpenCV libraries to detect and classify human activities in a video, specifically focusing on push-up, pull-up, or other activities.

## Features

- Real-time detection of human activities in a video stream.
- Uses the MediaPipe library for accurate pose estimation.
- Implements OpenCV for video processing and visualization.
- Supports detection of push-up and pull-up positions.
- Easily extensible for detecting other activities.

## Requirements

- Python 3.7 or later
- MediaPipe library
- OpenCV library

## Installation

1. Clone the repository:
2. Install the required libraries using pip:

## Usage

1. Place the video file you want to analyze in the project directory.
2. Open the command line and navigate to the project directory.
3. Run the script using the following command.
   `python main.py`

## Customization

- To add support for detecting additional activities, you can modify the conditions in `main.py` file.
- You can tweak the threshold values and conditions for detecting specific poses or movements based on your requirements.
- For more advanced customization, refer to the MediaPipe and OpenCV documentation.

## Examples

- Copy the path of video you want to test.
- Paste the path in cv2.VideoCapture() function or change it to 0 if you want to capture camera livestream.
- Then run the python script.

## License

This project is licensed under the [MIT License](LICENSE).
