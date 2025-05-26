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

## Progress

### 25% Report
- Finetuned a pretrained model YOLOv8s with 7k images of rock, paper, scissor
- Successfully detected hand symbols on live cam, albeit moderate detection
- Trained again with more images (15k+), produced more strong detection. Will using this model.
