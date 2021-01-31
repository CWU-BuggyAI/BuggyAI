import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import os
import numpy as np
import pathlib

def runPrediction(img_name):
    print('Loading model...')
    model = tf.keras.models.load_model('models/buggyAI_alpha.h5')
    print('Model loaded successfully')

        #img_path = 'static/uploads/' + img_name
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
        # decode the results into a list of tuples (class, description, probability)
        # (one such list for each sample in the batch)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


    print("donezoooo")


        
"""
    if os.path.exists(os.path.join(os.getcwd(), img_name)):
        os.remove(os.path.join(os.getcwd(), img_name))

    # trained_model.summary()
"""
