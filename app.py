from flask import render_template, jsonify, Flask, request, make_response
import os
import io
import numpy as np
from PIL import Image
from keras.models import model_from_json

app = Flask(__name__)

SKIN_CLASSES = {
    0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular skin lesion'
}

# Set the absolute paths to the model files dynamically based on the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_json_path = os.path.join(script_dir, 'model.json')
model_weights_path = os.path.join(script_dir, 'model.h5')

@app.route('/')
def index():
    return render_template('detect.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    return render_template('dashboard.html')

def findMedicine(pred):
    if pred == 0:
        return "fluorouracil"
    elif pred == 1:
        return "Aldara"
    elif pred == 2:
        return "Prescription Hydrogen Peroxide"
    elif pred == 3:
        return "fluorouracil"
    elif pred == 4:
        return "fluorouracil (5-FU):"
    elif pred == 5:
        return "fluorouracil"
    elif pred == 6:
        return "fluorouracil"

@app.route('/detect', methods=['POST'])
def detect():
    json_response = {}
    if request.method == 'POST':
        try:
            file = request.files['file']
        except KeyError:
            return make_response(jsonify({
                'error': 'No file part in the request',
                'code': 'FILE',
                'message': 'file is not valid'
            }), 400)

        imagePil = Image.open(io.BytesIO(file.read()))

        # Save the image to a BytesIO object
        imageBytesIO = io.BytesIO()
        imagePil.save(imageBytesIO, format='JPEG')
        imageBytesIO.seek(0)

        # Load model architecture from JSON
        with open(model_json_path, 'r') as json_file:
            loaded_json_model = json_file.read()

        # Load model from JSON
        model = model_from_json(loaded_json_model)

        # Load model weights
        model.load_weights(model_weights_path)

        img = np.array(imagePil.resize((224, 224)))
        img = img.reshape((1, 224, 224, 3))
        img = img / 255.0

        prediction = model.predict(img)
        pred = np.argmax(prediction)
        disease = SKIN_CLASSES[pred]
        accuracy = prediction[0][pred]
        accuracy = round(accuracy * 100, 2)
        medicine = findMedicine(pred)

        json_response = {
            "detected": False if pred == 2 else True,
            "disease": disease,
            "accuracy": accuracy,
            "medicine": medicine,
            "img_path": file.filename,
        }

        return make_response(jsonify(json_response), 200)
    else:
        return render_template('detect.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
