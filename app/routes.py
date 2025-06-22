import base64
import io
from flask import request, jsonify
from PIL import Image
from ultralytics import YOLO  # Make sure ultralytics is installed

model = YOLO('D:/Kuliah/CV/Project/cv-project/app/best.pt')  # Use your model path

from flask import request, jsonify, render_template, session

from app import app

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/play')
def play():
    return render_template('play.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    image_data = data['image'].split(',')[1]  # Remove data:image/png;base64,
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((640, 480))  # Resize to 640x480

    # Load YOLOv8 model (you can load once globally for efficiency)

    # Run inference
    results = model(image, conf=0.5)

    # Example: return detected class names
    detected = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            detected.append(model.names[cls])

    # Make a random pick from AI to battle with the user
    import random
    if detected:
        ai_choice = random.choice(['Scissors', 'Rock', 'Paper'])
        detected.append(f"AI chose: {ai_choice}")
    else:
        detected.append("No objects detected")

    if 'score' not in session:
        session['score'] = {'user': 0, 'ai': 0, 'tie': 0}

    # A simple Scissors, Rock, Paper game logic
    user_choice = detected[0]
    if user_choice not in ['Scissors', 'Rock', 'Paper']:
        return jsonify({'error': 'Invalid user choice. Please choose Scissors, Rock, or Paper.'}), 400
    
    if user_choice == ai_choice:
        result = "It's a tie!"
        session['score']['tie'] += 1
    elif (user_choice == 'Rock' and ai_choice == 'Scissors') or \
         (user_choice == 'Scissors' and ai_choice == 'Paper') or \
         (user_choice == 'Paper' and ai_choice == 'Rock'):
        result = "You win!"
        session['score']['user'] += 1
    else:
        result = "You lose!"
        session['score']['ai'] += 1
        
    session['score'] = dict(session['score'])

    return jsonify({
        'detected': detected, 
        'ai_choice': ai_choice, 
        'result': result, 
        'score': session['score']
    })