from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import io
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_model.pth")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using device: {DEVICE}")

def load_model():
    import clip
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    
    class DarkPatternClassifier(nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.clip_model = clip_model
            self.classifier = nn.Linear(512, 2)
        
        def forward(self, x):
            features = self.clip_model.encode_image(x)
            return self.classifier(features)
    
    classifier = DarkPatternClassifier(model)
    
    if os.path.exists(MODEL_PATH):
        classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully")
    else:
        print(f"Model not found at {MODEL_PATH}")
    
    classifier.to(DEVICE)
    classifier.eval()
    return classifier, preprocess

model, preprocess = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided', 'success': False}), 400
        
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            dark_prob = float(probs[0][1] * 100)
            is_dark_pattern = dark_prob > 50
        
        return jsonify({
            'success': True,
            'is_dark_pattern': is_dark_pattern,
            'dark_probability': round(dark_prob, 1),
            'clean_probability': round(100 - dark_prob, 1)
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(DEVICE)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
