from flask import Flask, request, jsonify, render_template_string
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from flask_cors import CORS
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

# Load your trained model
model = load_model('inceptionv3_bat_classification_model.keras')  # Update this with your model's path

# Define the class labels
class_labels = ['English Willow', 'Kashmir Willow', 'Other']

# Enhanced HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Prediction Tool</title>
    <style>
        body {
            font-family: 'Roboto', Arial, sans-serif;
            text-align: center;
            background: linear-gradient(135deg, #74ebd5, #ACB6E5);
            color: #333;
            margin: 0;
            padding: 0;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 20px;
            color: #8B4513;
            text-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
        }
        #container {
            width: 90%;
            max-width: 400px;
            background: #fff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            text-align: center;
        }
        input[type="file"] {
            margin: 10px auto;
            display: block;
            background: #f4f4f4;
            padding: 10px;
            border: 2px dashed #ddd;
            border-radius: 8px;
            cursor: pointer;
            transition: border-color 0.3s;
        }
        input[type="file"]:hover {
            border-color: #74ebd5;
        }
        button {
            margin: 5px;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: bold;
            color: #fff;
            background: linear-gradient(135deg, #74ebd5, #ACB6E5);
            cursor: pointer;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.3);
        }
        #progress-bar-container {
            display: none;
            margin-top: 20px;
            width: 100%;
            height: 10px;
            background: #f4f4f4;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        #progress-bar {
            height: 100%;
            background: linear-gradient(135deg, #74ebd5, #ACB6E5);
            width: 0%;
            transition: width 0.3s ease;
        }
        #result-container {
            margin-top: 20px;
        }
        img {
            max-width: 100%;
            max-height: 200px;
            border-radius: 10px;
            margin-top: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        #label {
            font-size: 1.2rem;
            font-weight: bold;
            color: #333;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div id="container">
        <h1>Cricket bat Willow Prediction Tool</h1>
        <input type="file" id="imageUpload" name="image" accept="image/*" required>
        <button onclick="submitImage()">Submit</button>
        <button onclick="resetForm()">Reset</button>
        <div id="progress-bar-container">
            <div id="progress-bar"></div>
        </div>
        <div id="result-container">
            <img id="result-image" src="" alt="" style="display:none;">
            <p id="label"></p>
        </div>
    </div>

    <script>
        function submitImage() {
            const imageInput = document.getElementById('imageUpload');
            const formData = new FormData();
            formData.append('image', imageInput.files[0]);

            // Show progress bar
            const progressBarContainer = document.getElementById('progress-bar-container');
            const progressBar = document.getElementById('progress-bar');
            progressBarContainer.style.display = 'block';
            progressBar.style.width = '0%';

            let progress = 0;
            const interval = setInterval(() => {
                if (progress < 90) {
                    progress += 10;
                    progressBar.style.width = progress + '%';
                } else {
                    clearInterval(interval);
                }
            }, 300);

            // Send POST request to Flask API
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                clearInterval(interval);
                progressBar.style.width = '100%'; // Complete the progress bar

                if (data.error) {
                    document.getElementById('label').innerText = data.error;
                    document.getElementById('result-image').style.display = "none";
                } else {
                    // Show the uploaded image
                    const file = imageInput.files[0];
                    const reader = new FileReader();
                    reader.onload = function(event) {
                        document.getElementById('result-image').src = event.target.result;
                        document.getElementById('result-image').style.display = "block";
                    };
                    reader.readAsDataURL(file);

                    // Show the predicted label
                    document.getElementById('label').innerText = 'Predicted Class: ' + data.predicted_class;
                }

                // Hide the progress bar after completion
                setTimeout(() => {
                    progressBarContainer.style.display = 'none';
                }, 500);
            })
            .catch(error => {
                clearInterval(interval);
                progressBar.style.width = '100%';
                console.error('Error:', error);
                document.getElementById('label').innerText = 'Error occurred while processing the image.';
                document.getElementById('result-image').style.display = "none";

                // Hide the progress bar after failure
                setTimeout(() => {
                    progressBarContainer.style.display = 'none';
                }, 500);
            });
        }

        function resetForm() {
            document.getElementById('imageUpload').value = ''; // Clear the file input
            document.getElementById('result-image').src = ''; // Clear the image preview
            document.getElementById('result-image').style.display = 'none';
            document.getElementById('label').innerText = ''; // Clear the result label
            const progressBarContainer = document.getElementById('progress-bar-container');
            progressBarContainer.style.display = 'none'; // Hide the progress bar
            document.getElementById('progress-bar').style.width = '0%'; // Reset progress bar
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found in request'}), 400

    img_file = request.files['image']

    # Use the stream attribute for the image
    img = image.load_img(BytesIO(img_file.read()), target_size=(299, 299))

    # Process the image as usual
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Convert the image to grayscale for GLCM
    gray_image = rgb2gray(img_array[0])

    # Compute GLCM features (use the same parameters as used during training)
    glcm_matrix = graycomatrix((gray_image * 255).astype('uint8'), distances=[1], angles=[0], symmetric=True,
                               normed=True)

    # Use only 4 GLCM features (like during training)
    glcm_contrast = graycoprops(glcm_matrix, 'contrast').flatten()
    glcm_homogeneity = graycoprops(glcm_matrix, 'homogeneity').flatten()
    glcm_energy = graycoprops(glcm_matrix, 'energy').flatten()
    glcm_correlation = graycoprops(glcm_matrix, 'correlation').flatten()

    # Combine GLCM features into one feature vector (4 features)
    glcm_features = np.concatenate([glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation])
    glcm_features = np.expand_dims(glcm_features, axis=0)  # Expand dims to match batch size

    # Predict class with both image and GLCM features
    prediction = model.predict([img_array, glcm_features])
    predicted_class = np.argmax(prediction, axis=1)

    # Return the predicted class as a JSON response
    return jsonify({'predicted_class': class_labels[predicted_class[0]]})


if __name__ == '__main__':
    app.run(debug=True)
