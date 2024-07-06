from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Kayıtlı modeli yükle
model = load_model("vgg16_model.keras")

# Sınıf adlarını belirleyin 
class_names = ["cat", "dog"]  

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)
            result = classify_image(img_path)
            image_url = url_for('static', filename='uploads/' + file.filename)
            return render_template("index.html", result=result, image_url=image_url)
    return render_template("index.html")

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(53, 53))  # Resim boyutunu 53x53 olarak ayarla
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizasyon

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    return f"{predicted_class} ({confidence:.2f}%)"

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
