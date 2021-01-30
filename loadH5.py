import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import os
import numpy as np

def runPrediction(img_name):
    
    print('Loading model...')
    trained_model = tf.keras.models.load_model('models/mobilenetv2.h5')
    print('Model loaded successfully')

    #img_path = 'static/uploads/' + img_name

    img = image.load_img(img_name, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = trained_model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])

    if os.path.exists(os.path.join(os.getcwd(), img_name)):
        os.remove(os.path.join(os.getcwd(), img_name))

    # trained_model.summary()