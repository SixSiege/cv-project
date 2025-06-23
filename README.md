# Computer Vision Project
### Rock Paper Scissor Game using YOLOv8 for Object Detection
Name: M. Amar Fauzan

Class: 4 TI D Information Technology

## Background
Rock Paper Scissor is a traditional game played by many people in the world, where the game is played with 2 players. Both players sends out one of the three hand symbols depicting Rock, Paper and Scissor. There are game logics to determine winner with different hand symbols combination. The game is very simple and very easy to understand, but required to have 2 players. The game can't be played if there's only 1 player. That's Computer Vision comes in. With the power of Deep Learning, player can play a Rock Paper Scissor with an AI, just using a smartphone. The AI will predict player's hand symbol after several second and pick a random hand symbol in the system, making it 1 player is possible by AI as a second player.

## Usage
Human player will start the playing session by pressing a button. A countdown will appeared on screen with webcam/front camera turned on. On the last countdown, human player will choose a Scissor, a Rock or a Paper using their hand. For some time, the camera captures the hand symbol and quickly categorized it using YOLO. From the AI side, system will randomly selected 1 of the options. Using code logics, the result of the game will be displayed to set the winner of the game.

## Tools & Technologies
- VS Code
- Flask
- Python
- Ultralytics

## Progress

### 25% Report
- Finetuned a pretrained model YOLOv8s with 7k images of rock, paper, scissor
- Successfully detected hand symbols on live cam, albeit moderate detection
- Trained again with more images (15k+), produced more strong detection. Will using this model.

### 75% Report
- Implement a base code Flask for detecting and classifying image
- Will implement new features

### 100% Report
- Implemented a scoring feature.
- Making the appearance better

## Instruction
1. Download dataset from roboflow
```python
from roboflow import Roboflow
rf = Roboflow(api_key=api_key)
project = rf.workspace("roboflow-58fyf").project("rock-paper-scissors-sxsw")
version = project.version(10)
dataset = version.download("yolov8")
```
2. Install libraries
```python
pip install opencv-python matplotlib numpy
```
3. Train model
```python
from ultralytics import YOLO

# Load a pretrained YOLOv8s model
model = YOLO('yolov8s.pt')  # Small version

# Train the model
results = model.train(
    data='rock-paper-scissors-10/data.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    device='0',  # '0' for GPU, 'cpu' for CPU
    name='rps_yolov8n_first_run'
)
```
4. Evaluate model
```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('best.pt')

# Evaluate
metrics = model.val()  # Will use the validation set from data.yaml
print(f"mAP@0.5: {metrics.box.map:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map75:.3f}")
```
5. Run model on camera
```python
import cv2
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Open webcam
cap = cv2.VideoCapture(0)
frame_count = 0  # Initialize frame counter

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break
  
  frame_count += 1  # Increment frame counter
  
  # Process every 8th frame
  if frame_count % 4 == 0:
    # Run inference
    results = model(frame, conf=0.5)  # confidence threshold of 0.5
    
    # Visualize
    annotated_frame = results[0].plot()
    cv2.imshow('Rock-Paper-Scissors Detection', annotated_frame)
  
  if cv2.waitKey(1) == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
```
6. Run model on test pictures
```python
import os
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Get and shuffle test images
test_dir = 'C:/cv-project/rock-paper-scissors-14/test/images'
images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg')]
random.shuffle(images)

# Create figure
plt.figure(figsize=(20, 15))

# Process 4 images
for idx, img_path in enumerate(images[:4], 1):
    # Run prediction
    results = model(img_path, conf=0.5)

    # Get annotated image
    annotated_img = results[0].plot()[..., ::-1]  # BGR to RGB

    # Add subplot
    plt.subplot(2, 2, idx)
    plt.imshow(annotated_img)

    # Add title with detected classes
    detections = results[0].boxes
    if len(detections) > 0:
        class_ids = detections.cls.tolist()
        conf_scores = detections.conf.tolist()
        class_names = [results[0].names[int(c)] for c in class_ids]
        title = ", ".join([f"{n} ({c:.2f})" for n, c in zip(class_names, conf_scores)])
        plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
```
## Instructions (web app)
1. Create venv
```
python -m venv .venv
```
2. Enter the new venv
```powershell
.venv\Scripts\Activate
```
3. Install requirements
```
pip install -r requirement.txt
```
4. Set flask app
```
set FLASK_APP=app
```
5. Run flask app (will run with local cert and debug env)
```
flask run --debug --cert=adhoc
```
