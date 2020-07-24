import tensorflow as tf
import os
import numpy as np
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
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
        md = tf.keras.models.load_model('model_0.93')
        image_dims = 160
        DIR = os.path.join('static/uploads', filename)
        test_image = load_img(DIR,target_size=(image_dims, image_dims))
        # plt.show(test_image)
        
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255
        result = md.predict(test_image)
        print(result[0])
        maxValue= -1
        for i in result[0]:
                if i > maxValue:
                        maxValue = i

        print("max: {}".format(maxValue))
        return result[0] * 100
