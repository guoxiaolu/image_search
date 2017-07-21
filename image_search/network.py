#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
os.environ['KERAS_BACKEND']='theano'
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
# from keras.applications import VGG16
import keras.backend as K
K.set_image_dim_ordering('th')
import numpy as np
from get_label import get_labels

def VGG_16(weight_path=None):
    """VGG 16 model

    """

    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten(name="flatten"))
    model.add(Dense(4096, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, name='dense_3'))
    model.add(Activation("softmax", name="softmax"))

    if weight_path:
        model.load_weights(weight_path)
    return model

def load_deep_model(weight_path=None):
    """Pop the last softmax layer, and use sigmoid layer instead

    """
    base_model = VGG_16(weight_path)

    model = Model(input=base_model.input, output=base_model.get_layer('dense_3').output)

    return model

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-1.0 * z))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def predict_model(model, img, labels_id, labels_name):
    """Model prediction output
    """

    out = model.predict(img)
    out = out.flatten()
    label = softmax(out)
    top_3 = np.argsort(label)[::-1][:3]
    top_3_prob = label[top_3]
    label_filter = get_labels(labels_id, labels_name, top_3, top_3_prob)
    out = sigmoid(out)
    return out, label_filter
