#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:35:56 2017

@author: Lavector
"""

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import backend as K
from data import img_gen


def VGG_16(weights_path=None):
    model = Sequential()

    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten(name="flatten"))
    model.add(Dense(4096, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, name='dense_3'))
    model.add(Activation("softmax",name="softmax"))

    if weights_path:
        model.load_weights(weights_path)
    return model
    
def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)
    
def triplet_loss(vects):
    anchor, pos, neg = vects
    loss = K.maximum(0, (1-K.sum((anchor - neg)**2, axis=-1, keepdims=True)
                        + K.sum((anchor - pos)**2, axis=-1, keepdims=True)))
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
    
def create_base_network(weights_path = None):
    base_model = VGG_16(weights_path)
    
    pop_layer(base_model)
       
    x = base_model.outputs

    for i, layer in enumerate(base_model.layers):
        if i <= 30:
            layer.trainable = False
        
    x = Activation(activation = 'sigmoid')(x)
        
    model = Model(input=base_model.input, output=x)
    
    return model

vgg16_weights_path = '/Users/Lavector/code/convnets-keras-master/weights/vgg16_weights.h5'
img_rows, img_cols = 224, 224

if K.image_dim_ordering() == 'th':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)
    
input_a = Input(shape = input_shape)
input_p = Input(shape = input_shape)
input_n = Input(shape = input_shape)
base_network = create_base_network(vgg16_weights_path)

processed_a = base_network(input_a)
processed_p = base_network(input_p)
processed_n = base_network(input_n)

loss = Lambda(triplet_loss, output_shape=triplet_loss_output_shape)([processed_a, processed_p, processed_n])

model = Model(input=[input_a, input_p, input_n], output = loss)

model.compile(loss=identity_loss, optimizer=SGD())

model.summary()

train_path = '/Users/Lavector/code/cbir_tl/train.txt'
test_path = '/Users/Lavector/code/cbir_tl/test.txt'
#val_path = '/Users/Lavector/code/cbir_tl/val.txt'
img_path = '/Users/Lavector/dataset/VOC2012/JPEGImages'
nb_epoch = 50000

#anchor_val, pos_val, neg_val = img_all(val_path, img_path)
#print 'read validation set! Image size: %d'%(len(anchor_val))
for e in range(nb_epoch):
    print("epoch %d" % e)
    for anchor, pos, neg in img_gen(train_path, img_path, 1):
        model.fit([anchor, pos, neg], np.ones(anchor.shape[0]),
#                  validation_data=([anchor_val, pos_val, neg_val], np.ones(anchor_val.shape[0])),
                  batch_size=1,
                  nb_epoch=1,
                  verbose=1,
                  shuffle=True)
    if e%1000 == 0 or e == (nb_epoch-1):
        model.save('/users/lavector/tmp/weights_%04d.hdf5'%(e))

# model.load_weights('/users/lavector/tmp/weights_00.hdf5')
# vgg_model = model.layers[-2]
# print [layer.name for layer in vgg_model.layers]