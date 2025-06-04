from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from flask_cors import CORS  # type: ignore
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = load_model('my_model.h5')  # Load your trained model

@app.route("/predict", methods=["POST"])
def predict():
    img_file = request.files['image']
    os.makedirs("uploads", exist_ok=True)
    img_path = os.path.join("uploads", img_file.filename)
    img_file.save(img_path)

    # Preprocess the image
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = img_array.reshape(1, 28, 28, 1) / 255.0

    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))

    return jsonify({"prediction": predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
