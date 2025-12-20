# Sign Language Interpreter using Deep Learning

A sign language interpreter using live video feed from the camera. This project uses computer vision and machine learning techniques to recognize hand gestures and translate them into text.

## Features

- Real-time ASL (American Sign Language) alphabet recognition
- Gesture recognition for numbers and basic commands
- Text-to-speech functionality
- Multiple operation modes:
  - Alphabet Mode: Recognize ASL letters A-Z
  - Text Mode: General gesture recognition
  - Calculator Mode: Mathematical operations using gestures

## Technologies Used

- Python
- OpenCV for computer vision
- MediaPipe for hand tracking
- TensorFlow/Keras for machine learning
- pyttsx3 for text-to-speech

## Setup

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the main application:
   ```
   cd Code
   python final.py
   ```

## Usage

- Press 'a' for Alphabet Mode
- Press 't' for Text Mode
- Press 'c' for Calculator Mode
- Press 'q' to quit
- Press 'v' to toggle voice output

## Requirements

- Python 3.7+
- Webcam or camera device