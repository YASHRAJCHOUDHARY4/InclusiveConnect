import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

app = Flask(__name__)
CORS(app)  # Allows the frontend to talk to backend easily

# --- LOAD SECRETS ---
VISION_ENDPOINT = os.environ['VISION_ENDPOINT']
VISION_KEY = os.environ['VISION_KEY']
CUSTOM_ENDPOINT = os.environ['CUSTOM_ENDPOINT']
CUSTOM_KEY = os.environ['CUSTOM_KEY']
PROJECT_ID = os.environ['PROJECT_ID']
PUBLISH_NAME = "Iteration2"  # This must match what you named your model when publishing


# --- HOME PAGE ---
@app.route('/')
def home():
    return render_template('index.html')


# --- FEATURE 1: DESCRIBE SCENE (For Visually Impaired) ---
@app.route('/describe', methods=['POST'])
def describe():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        img_data = file.read()

        # Connect to Azure Vision
        client = ImageAnalysisClient(endpoint=VISION_ENDPOINT,
                                     credential=AzureKeyCredential(VISION_KEY))

        # Analyze
        result = client.analyze(image_data=img_data,
                                visual_features=[VisualFeatures.CAPTION])

        description = result.caption.text if result.caption else "No description found."
        return jsonify({'message': description})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- FEATURE 2: SIGN LANGUAGE TRANSLATOR (For Non-Verbal) ---
@app.route('/sign', methods=['POST'])
def sign_language():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        # Custom Vision requires the file object.
        file.seek(0)

        # Connect to Custom Vision
        credentials = ApiKeyCredentials(
            in_headers={"Prediction-key": CUSTOM_KEY})
        predictor = CustomVisionPredictionClient(CUSTOM_ENDPOINT, credentials)

        # Predict
        results = predictor.classify_image(PROJECT_ID, PUBLISH_NAME, file)

        # Find best prediction
        best_pred = max(results.predictions, key=lambda p: p.probability)

        # --- DEBUGGING PRINT (Look at your Replit Console/Shell to see this) ---
        print(
            f"I saw: {best_pred.tag_name} with confidence: {best_pred.probability:.2f}"
        )

        # --- UPDATED LOGIC: Lower Threshold to 15% (0.15) ---
        if best_pred.probability > 0.15:
            msg = f"Sign Detected: {best_pred.tag_name}"
            # Add a warning if confidence is low but acceptable
            if best_pred.probability < 0.5:
                msg += f" (Probability : {best_pred.probability*100:.0f}%)"

            return jsonify({
                'message': msg,
                'confidence': f"{best_pred.probability*100:.2f}%"
            })
        else:
            return jsonify({
                'message':
                f"Not sure. Best guess was {best_pred.tag_name} ({best_pred.probability*100:.0f}%)"
            })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
