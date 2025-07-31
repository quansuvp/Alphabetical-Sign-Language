# Alphabetical Sign Language Recognition

This project uses MediaPipe to extract hand landmarks from video frames and an SVM classifier to recognize basic sign language gestures.

## Features

- Extract 3D hand landmarks using MediaPipe's `HandLandmarker`.
- Classify basic hand signs using an SVM model (e.g., A, B, C, etc.).
- Accuracy up to 97.5% on validation set.

## Setup Instructions

### 1. Install Requirements

```pip install -r requirements.txt```
### 2. Run
``` python main.py ```
