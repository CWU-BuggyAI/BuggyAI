from flask import Flask, render_template, jsonify, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import os
from PIL import Image
import numpy as np
import pathlib

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Loading the ML model
print("Loading ML Model")
model = tf.keras.models.load_model('models/buggyAI_alpha.h5')

@app.route('/', methods=["GET", "POST"])
def pred():
    if request.method == 'POST':
        print('[----- Buggy AI -----]')

        imgFile = request.files['file']
        print('Image recieved from request')

        filename = secure_filename(imgFile.filename)
        print('Checked filename: ', filename)

        # print('converting image from stream with PIL')
        # img = Image.open(imgFile.stream)

        # print(jsonify({'msg': 'success', 'size': [img.width, img.height]}))
        imgFile.save(os.path.join(os.getcwd(), imgFile.filename))

        print("Running image classification..")
        result = runPrediction(filename)
        print("Image Classified, returning result")

        return result
    
    return render_template('index.html')

# @app.route('/', methods=["GET"])
# def home():

#     resp = ('Welcome to the BuggyAI API, this is a test of the GET request. '
#     'Our application mainly handles POST requests. In order to run a image classification '
#     'with our model you must pass a .png or .jpg image in the form of a POST request.')

#     print(resp)
#     return render_template('index.html')

# Run prediction analysis given image
def runPrediction(img_name):
    # img_array = tf.keras.preprocessing.image.img_to_array(pil_img)

    img_height = 180
    img_width = 180

    class_names = ['aphids', 'red-spider-mites', 'thrips']

    test_data = img_name
    test_dir = pathlib.Path(test_data)

    img = keras.preprocessing.image.load_img(
        test_dir, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    if os.path.exists(os.path.join(os.getcwd(), img_name)):
        os.remove(os.path.join(os.getcwd(), img_name))

    pred_str = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    return pred_str

if __name__ == '__main__':
    app.run()