import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import random
import cv2
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing import image

def predict(filename):
        md = tf.keras.models.load_model('static/my_model.h5')
        image_dims = 160
        DIR = os.path.join('static/images', filename)
        test_image = load_img(DIR,target_size=(image_dims, image_dims))

        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255
        result = md.predict(test_image)

        maxValue= -1
        for i in result[0]:
                if i > maxValue:
                        maxValue = i


        print(result[0] * 100)
        return result[0] * 100

app = Flask(__name__,template_folder='templates')
app.config['IMAGE_UPLOADS'] = "static/images"
app.config['ALLOWED_IMAGE_EXTENSIONS'] = ['JPEG', 'JPG', 'PNG']

def allowed_image(filename):
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            if image.filename == "":

                return redirect(request.url)

            if not allowed_image(image.filename):

                return redirect(request.url)

            else:
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

            prd = predict(filename)
            os.remove(os.path.join(app.config["IMAGE_UPLOADS"],filename))
            return render_template('index.html',cnv=prd[0], dme=prd[1], drusen=prd[2], normal=prd[3], prediction=True)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
