from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import io
import os

model = tf.keras.models.load_model("./ANNModel.h5")

app = Flask(__name__)

def preprocess_image(image):
    """Convert uploaded image to MNIST format (28x28, grayscale, normalized)"""
    image = image.convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors (MNIST digits are white-on-black)
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image = np.array(image) / 255.0  # Normalize (0-1)
    image = image.reshape(1, 28, 28, 1)  # Add batch + channel dimensions
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    return jsonify({"prediction": predicted_digit, "confidence": round(confidence, 2)})
#
# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)