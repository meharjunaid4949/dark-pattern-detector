from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import io
import os

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Looking for model at: {MODEL_PATH}")

# Load CLIP model and processor from transformers
def load_model():
    try:
        # Load base CLIP model and processor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("✅ Base CLIP model loaded successfully")
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        raise e
    
    # Add a classification layer
    class DarkPatternClassifier(torch.nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.clip_model = clip_model
            self.classifier = torch.nn.Linear(768, 2)
        
        def forward(self, pixel_values):
            features = self.clip_model.get_image_features(pixel_values=pixel_values)
            return self.classifier(features)
    
    classifier = DarkPatternClassifier(model)
    
    # Load your fine-tuned weights if they exist
    if os.path.exists(MODEL_PATH):
        try:
            classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
            print("✅ Fine-tuned model weights loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load fine-tuned weights: {e}")
            print("Starting with base CLIP model only")
    else:
        print(f"⚠️ No fine-tuned model found at {MODEL_PATH}")
        print("Starting with base CLIP model only")
    
    classifier.to(DEVICE)
    classifier.eval()
    return classifier, processor

# Load the model
model, processor = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided', 'success': False}), 400
        
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Process the image using the processor
        inputs = processor(images=image, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(DEVICE)
        
        # Inference
        with torch.no_grad():
            logits = model(pixel_values)
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
    port = int(os.environ.get('PORT', 10000))
    print(f"Starting server on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
