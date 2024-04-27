import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('model_new_brain.h5')  # Load your updated model trained for 4 classes

@app.route('/')
def home():
    return render_template('home2.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is included in the request
        if 'file' not in request.files:
            return render_template('home2.html', prediction_text="No file uploaded")

        file = request.files['file']
        # Check if file name is empty
        if file.filename == '':
            return render_template('home2.html', prediction_text="No file selected")

        # Open and preprocess the image
        img = Image.open(file)
        img = img.resize((64, 64))  # Resize image to match model input shape
        img = np.array(img) / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 64, 64, 3)

        # Make prediction
        prediction = model.predict(img)

        # Convert prediction to human-readable label
        class_labels = ['Glioma', 'Meningioma', 'Notumor', 'Pituitary']
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]

        # Redirect to result page with the predicted class
        return redirect(url_for('result', prediction_text=predicted_class))

    except Exception as e:
        return render_template('home2.html', prediction_text="Error occurred during prediction: {}".format(str(e)))

@app.route('/result')
def result():
    prediction_text = request.args.get('prediction_text')
    if prediction_text:
        return render_template('result.html', prediction_text=prediction_text)
    else:
        return render_template('result.html', prediction_text="No prediction result")

if __name__ == "__main__":
    app.run(debug=True)
