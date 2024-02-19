from flask import Flask, jsonify, request
from tensorflow import keras
import numpy as np
from PIL import Image
from io import BytesIO
import requests

app = Flask(__name__)

# Load the pre-trained model
model = keras.models.load_model('model.h5')

# Define the class names
class_names = ["Tuberculosis", "healthy", "latent-tb", "uncertain-tb"]

def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))  # Resize image to match the input size used during training
    image_array = np.asarray(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request contains a file or a URL
        if 'file' in request.files:
            # Get the image file from the request
            image_file = request.files['file']
            # Read image bytes and preprocess
            image_bytes = image_file.read()
        elif 'url' in request.form:
            # Get the image URL from the request
            image_url = request.form['url']
            # Download image from the URL and preprocess
            response = requests.get(image_url)
            response.raise_for_status()  # Check for HTTP errors
            image_bytes = response.content
        else:
            raise ValueError("Neither file nor URL provided in the request.")

        image_array = preprocess_image(image_bytes)

        # Make prediction
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]

        # Return the result as JSON
        return jsonify({'class': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
