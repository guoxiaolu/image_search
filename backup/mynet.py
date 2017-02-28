#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam, SGD
from keras import backend as K


def VGG_16(weights_path=None):
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

    if weights_path:
        model.load_weights(weights_path)
    return model

def triplet_loss(vects, alpha=1):
    anchor, pos, neg = vects
    loss = K.maximum(0, alpha - K.sum((anchor - neg) ** 2, axis=-1, keepdims=True)
                     + K.sum((anchor - pos) ** 2, axis=-1, keepdims=True))
    return loss

def triplet_loss_output_shape(shapes):
    shape1, shape2, shape3 = shapes
    return (shape1[0], 1)

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

def create_base_network(weights_path=None):
    base_model = VGG_16(weights_path)

    pop_layer(base_model)

    x = base_model.outputs

    for i, layer in enumerate(base_model.layers):
        if i <= 30:
            layer.trainable = False

    x = Activation(activation='sigmoid')(x)

    model = Model(input=base_model.input, output=x)

    return model

def VGG_16_tl(weights_path=None):
    if K.image_dim_ordering() == 'th':
        input_shape = (3, 224, 224)
    else:
        input_shape = (224, 224, 3)
    input_a = Input(shape=input_shape)
    input_p = Input(shape=input_shape)
    input_n = Input(shape=input_shape)
    base_network = create_base_network('/Users/Lavector/code/convnets-keras-master/weights/vgg16_weights.h5')

    processed_a = base_network(input_a)
    processed_p = base_network(input_p)
    processed_n = base_network(input_n)

    loss = Lambda(triplet_loss, output_shape=triplet_loss_output_shape)([processed_a, processed_p, processed_n])

    model = Model(input=[input_a, input_p, input_n], output=loss)

    if weights_path != None:
        model.load_weights(weights_path)
    return model