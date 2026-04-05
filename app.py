from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model("model.h5")

# IMPORTANT: match training order (alphabetical)
classes = ["Early_Blight", "Healthy", "Late_Blight"]

def predict(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    return classes[np.argmax(prediction)]

# 🔥 NEW: pesticide logic
def pesticide_advice(result):
    if result == "Healthy":
        return "No pesticide needed"
    elif result == "Early_Blight":
        return "Low pesticide recommended"
    elif result == "Late_Blight":
        return "High pesticide required"

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    advice = None

    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file)

        result = predict(img)
        advice = pesticide_advice(result)

    return render_template("index.html", result=result, advice=advice)

if __name__ == "__main__":
    app.run(debug=True)