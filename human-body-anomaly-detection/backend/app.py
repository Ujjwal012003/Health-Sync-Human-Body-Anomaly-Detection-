# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import os


app = Flask(__name__)
CORS(app)

MODEL_PATH = "./model/anomaly_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to make prediction
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    condition = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)
    return condition, confidence

# API Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)
    condition, confidence = predict_image(img_path)
    os.remove(img_path)
    return jsonify({'prediction': str(condition), 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
