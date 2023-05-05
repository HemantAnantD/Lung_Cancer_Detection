from flask import *
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model('model.hdf5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    classes_dir = ["Adenocarcinoma","Large cell carcinoma","Normal","Squamous cell carcinoma"]
    img = Image.open(request.files['file'].stream).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = np.argmax(model.predict(img_array))
    result = classes_dir[pred]
    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)