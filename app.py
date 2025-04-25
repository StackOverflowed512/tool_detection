from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from tool_detector import OrthopedicToolDetector, analyze_orthopedic_tools

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file temporarily
    temp_path = os.path.join(UPLOAD_FOLDER, 'temp.jpg')
    file.save(temp_path)

    try:
        # Process the image using your existing tool detector
        _, tools = analyze_orthopedic_tools(
            image_path=temp_path,
            auto_calibrate=True,
            debug=False
        )
        
        # Clean up the temporary file
        os.remove(temp_path)
        
        return jsonify(tools)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)