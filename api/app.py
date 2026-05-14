from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Simple test endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(DEVICE)})

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    print("=== PREDICT ENDPOINT HIT ===")
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided', 'success': False}), 400
        
        file = request.files['image']
        print(f"Received file: {file.filename}")
        
        # Just return a mock response for now to test if endpoint works
        return jsonify({
            'success': True,
            'is_dark_pattern': False,
            'dark_probability': 25.0,
            'clean_probability': 75.0
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"Starting server on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
