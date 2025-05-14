import os
import cv2
import numpy as np
import pytesseract
import google.generativeai as genai
# import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import re
from feedback import compare_texts  # Import feedback module
# from tensorflow.keras.models import load_model
import io

# Initialize Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Set paths
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Models
yolo_model = YOLO("./model/dyslexic_yolov8s.pt")  # YOLO Model for dysgraphia detection
# mnist_model = load_model("./model/mnist_cnn.h5")  # CNN Model for digit recognition

# Initialize Google Gemini API
genai.configure(api_key="AIzaSyCzA8l46hWR0Kc0bAIwuQDJ6U1YUzraGvw")

# Set Tesseract OCR Path (Update for your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Image Preprocessing for OCR
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    processed_path = os.path.join(UPLOAD_FOLDER, "processed.jpg")
    cv2.imwrite(processed_path, binary)
    return processed_path

# Extract text using OCR
def extract_text(image_path):
    processed_image_path = preprocess_image(image_path)
    custom_config = r'--psm 4 --oem 3'
    text = pytesseract.image_to_string(Image.open(processed_image_path), config=custom_config)
    return text.strip()

# Text Cleaning
def clean_text(text):
    text = re.sub(r'[_=-]', ' ', text)  # Replace underscores and dashes with spaces
    text = re.sub(r'[^A-Za-z0-9.,!?\'\s]', '', text)  # Keep only letters, numbers, and basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Grammar Correction with Gemini
def correct_with_gemini(text):
    response = genai.GenerativeModel("gemini-1.5-pro").generate_content(f"Correct this text: {text}")
    return response.text

# Preprocess and predict digit from canvas
def preprocess_canvas(image):
    image = image.resize((28, 28)).convert("L")  # Resize and convert to grayscale
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.reshape(1, 28, 28, 1)  # Reshape for CNN model
    image = 1 - image  # Invert colors (white to black)
    return image

# Flask Route: Upload Image and Process
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Step 1: Detect Dysgraphia using YOLO
    results = yolo_model(image_path)
    detected_image = results[0].plot()
    detected_image_path = os.path.join(UPLOAD_FOLDER, "detected_" + file.filename)
    Image.fromarray(detected_image).save(detected_image_path)

    # Step 2: Extract Text using OCR
    extracted_text = extract_text(image_path)
    cleaned_text = clean_text(extracted_text)

    # Step 3: Correct Grammar using Gemini
    final_corrected_text = correct_with_gemini(cleaned_text)

    # Step 4: Compare Handwriting with Corrected Text and Provide Feedback
    feedback_result = compare_texts(cleaned_text, final_corrected_text)
    print(f"Feedback: {feedback_result['feedback']}")

    return jsonify({
        "detected_image_url": detected_image_path,
        "extracted_text": cleaned_text,
        "corrected_text": final_corrected_text,
        "similarity": feedback_result["similarity"],
        "feedback": feedback_result["feedback"]
    })

# Flask Route: Handle Canvas Digit Recognition
# @app.route("/predict_digit", methods=["POST"])
# def predict_digit():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))  # Read image from request
    processed_image = preprocess_canvas(image)  # Preprocess for CNN model

    # prediction = mnist_model.predict(processed_image)  # Get model prediction
    # predicted_digit = np.argmax(prediction)  # Find the highest probability digit

    # return jsonify({"predicted_digit": int(predicted_digit)})

# Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
