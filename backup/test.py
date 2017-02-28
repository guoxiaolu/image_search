#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:18:15 2017

@author: Lavector
"""

import os
# os.environ['THEANO_FLAGS'] = "device=gpu"

import numpy as np
import math
from keras.models import Model
from keras.layers import Activation
import glob
import os.path
import matplotlib.pyplot as plt
from imgxml import XMLGenerator, XMLParser
import time
from scipy.misc import imread, imresize, imsave
from scipy.spatial.distance import cosine, euclidean, hamming

from mynet import VGG_16_tl, VGG_16, pop_layer
from data import read_img
from tqdm import tqdm

OUTPUT_DIM = 1000
def load_myModel(weights_path):

    base_model = VGG_16(weights_path)

    pop_layer(base_model)

    x = base_model.outputs
    x = Activation(activation = 'sigmoid')(x)

    model = Model(input=base_model.input, output=x)

    return model

def load_myModel1(weights_path, layer_name = 'maxpooling2d_5'):
    base_model = VGG_16(weights_path)

    model = Model(input=base_model.input,
                                     output=Activation(activation='sigmoid')(base_model.get_layer(layer_name).output))

    return model

def load_partModel(weights_path):
    model = VGG_16_tl(weights_path)
    partModel = model.layers[-2]
    return partModel
    
def euclidean_distance(vects):
    x, y = vects
    return math.sqrt(np.sum((x - y)**2))
#    return euclidean(x, y)
    
def hamming_distance(vects):
    x, y = vects
    return np.sum(np.logical_xor(x>0.5, y > 0.5))
#    return hamming(x>0.5, y > 0.5)

def cosine_disatance(vects):
    x, y = vects
    # return (1 - np.dot(x, y) / (math.sqrt(np.sum(x**2)) * math.sqrt(np.sum(y**2))))[0]
    return (1 - np.dot(x, y) / (math.sqrt(np.sum(x**2)) * math.sqrt(np.sum(y**2))))
   # return cosine(x, y)

def distance_batch(anchor, batch, dist_calc):
    dist = []
    for ele in batch:
        dist.append(dist_calc([anchor, ele]))
    return dist

def search(anchor_path, search_path, network, weights_path, dist_type = 'cosine', display = False, feature_path = None, save_path = None):
    im1 = read_img(anchor_path, (224,224))
    im1 = im1.reshape((1, im1.shape[0], im1.shape[1], im1.shape[2]))

    if network == 'vgg_16_tl':
        model = load_partModel(weights_path)
    else:
        # model = load_myModel(weights_path)
        model = load_myModel1(weights_path)
    out1 = model.predict(im1)
    out1 = out1.flatten()
    labels = []
    if feature_path == None:
        if save_path == None:
            all_files, outs2 = get_feature(search_path, model, 'features.xml')
        else:
            all_files, outs2 = get_feature(search_path, model, save_path)
    else:
        all_files, outs2, labels = read_feature(feature_path)
    
    print 'predict finished!'
    if dist_type == 'euclidean':
        dist = distance_batch(out1, outs2, euclidean_distance)
    if dist_type == 'hamming':
        dist = distance_batch(out1, outs2, hamming_distance)
    if dist_type == 'cosine':
        dist = distance_batch(out1, outs2, cosine_disatance)

    dist_zip = dict(zip(all_files, dist))
    dist_zip = sorted(dist_zip.items(), lambda x, y: cmp(x[1], y[1]))

    print dist_zip
    if display:
        plt_anchor = plt.subplot(331)
        plt_anchor.imshow(imread(anchor_path))
        plt_result = plt.subplot(334)
        plt_result.imshow(imread(dist_zip[0][0]))
        plt_result = plt.subplot(335)
        plt_result.imshow(imread(dist_zip[1][0]))
        plt_result = plt.subplot(336)
        plt_result.imshow(imread(dist_zip[2][0]))
        plt_result = plt.subplot(337)
        plt_result.imshow(imread(dist_zip[3][0]))
        plt_result = plt.subplot(338)
        plt_result.imshow(imread(dist_zip[4][0]))
        plt_result = plt.subplot(339)
        plt_result.imshow(imread(dist_zip[5][0]))
        plt.show()

    return dist_zip, labels

def searchByModel(anchor_path, model, dist_type = 'cosine', feature_path = None):
    im1 = read_img(anchor_path, (224, 224))
    im1 = im1.reshape((1, im1.shape[0], im1.shape[1], im1.shape[2]))
           
    out1 = model.predict(im1)
    out1 = out1.flatten()
    if feature_path != None:
        all_files, outs2, _ = read_feature(feature_path)
    
    print 'predict finished!'
    start = time.time()
    if dist_type == 'euclidean':
        dist = distance_batch(out1, outs2, euclidean_distance)
    if dist_type == 'hamming':
        dist = distance_batch(out1, outs2, hamming_distance)
    if dist_type == 'cosine':
        dist = distance_batch(out1, outs2, cosine_disatance)
    print (time.time() - start)
    dist_zip = dict(zip(all_files, dist))
    dist_zip = sorted(dist_zip.items(), lambda x, y: cmp(x[1], y[1]))    
    
    return dist_zip
    
def read_feature(feature_path, feature_dim = OUTPUT_DIM):
    doc = XMLParser(feature_path)
    item_num = int(doc.getRootAttrib('num'))
    outs = np.zeros((item_num, feature_dim))
    all_files = []
    labels = []
    imgs = doc.getAllNodes('img')
    for i, img in enumerate(imgs):
        data = doc.getNodeValue(img, 'label')
        labels.append(data)
        data = doc.getNodeValue(img, 'path')
        all_files.append(data)
        data = doc.getNodeValue(img, 'feature')
        data_float = map(float, data.strip().split(','))
        outs[i] = np.array(data_float)  
    return all_files, outs, labels

def get_feature(search_path, model, save_path):
    all_files = glob.glob(os.path.join(search_path, '*.jpg'))
    
    doc = XMLGenerator(save_path)
    root = doc.createNode('imgs')
    doc.setNodeAttr(root, 'dataset', 'Caltech-101')
    doc.setNodeAttr(root, 'num', str(len(all_files)))
    doc.addNode(root)

    # outs = np.zeros((len(all_files), 1000))
    outs = np.zeros((len(all_files), OUTPUT_DIM))
    for i, name in enumerate(tqdm(all_files)):
        img = doc.createNode('img')
        doc.addNode(img, root)
        
        idx = doc.createNode('id')
        doc.setNodeValue(idx, str(i))
        doc.addNode(idx, img)
        
        label = doc.createNode('label')
        doc.setNodeValue(label, (os.path.basename(name)).split('-')[0])
        doc.addNode(label, img)
        
        path = doc.createNode('path')
        doc.setNodeValue(path, name)
        doc.addNode(path, img)
        
        feature = doc.createNode('feature')

        im = read_img(name, (224, 224))
        im = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
        out = model.predict(im)
        out = out.flatten()
        outs[i] = out

        doc.setNodeValue(feature, str(out.tolist())[1:-1])
        # doc.setNodeValue(feature, str(out[0].tolist())[1:-1])
        doc.addNode(feature, img)
    
    doc.genXml()
    return all_files, outs

def evaluation(anchor_path, network, weights_path, feature_path, top_n = 5, eval_save_path = None):
    all_files = glob.glob(os.path.join(anchor_path, '*.jpg'))
    total_num = len(all_files) * top_n * 1.0
    acc_num = 0
    if network != 'vgg_16_tl':
        model = load_myModel(weights_path)
        # model = load_myModel1(weights_path)
    else:
        model = load_partModel(weights_path)
    X = []
    Y = []
    for i, anchor in enumerate(all_files):
        dist_zip = searchByModel(anchor, model, dist_type = 'cosine', feature_path = feature_path)
        label_anchor = (os.path.basename(anchor)).split('-')[0]
        labels = []
        for n in xrange(500):
            # label = 0
            label = (os.path.basename(dist_zip[n][0])).split('-')[0]
            X.append(dist_zip[n][1])
            if label_anchor == label:
                Y.append(1)
            else:
                Y.append(0)
        if eval_save_path != None:
            dst = np.zeros((227*2, 227 * top_n, 3))
            img = imread(anchor, mode = 'RGB')
            img = imresize(img, (227, 227, 3))
            dst[:227,:227,:] = img
            basename = os.path.basename(anchor).split('.')
            basename0 = basename[0]
            extname = basename[1]
            name = basename0
        for n in xrange(top_n):
            # label = 0
            label = (os.path.basename(dist_zip[n][0])).split('-')[0]
            labels.append(label)
            if label_anchor == label:
                acc_num = acc_num + 1
            
            if eval_save_path != None:
                img_name = os.path.join('/Users/Lavector/testdata/imgset', os.path.basename(dist_zip[n][0]))
                img = imread(img_name, mode = 'RGB')
                # img = imread(dist_zip[n][0], mode = 'RGB')
                img = imresize(img, (227, 227, 3))
                dst[227:,n * 227:(n + 1) * 227,:] = img
                # name = name + '_%s(%.04f)'%(label, dist_zip[n][1])
                name = name + '_%.04f' % (dist_zip[n][1])
        
        if eval_save_path != None:
            if not os.path.exists(eval_save_path):
                os.mkdir(eval_save_path)
            # name = '%04d.jpg'%(i)
            name =os.path.join(eval_save_path, name+'.jpg')
            # name = os.path.join(eval_save_path, name + '.' + extname)
            imsave(name, dst)
        print anchor
        print '%d/%d, acc: %d, %.04f'%(i + 1, len(all_files), acc_num, acc_num / ((i + 1) * top_n * 1.0))
        print '%s:'%label_anchor
        print labels
    return acc_num / total_num, X, Y


    
eval_path = '/Users/Lavector/testdata/testset'
anchor_path = '/Users/Lavector/testdata/jd_test/TB22WXlexhmpuFjSZFyXXcLdFXa_!!1621502511.jpg'
search_path = '/Users/Lavector/testdata/imgset_01'
vgg16_weights_path = '/Users/Lavector/code/convnets-keras-master/weights/vgg16_weights.h5'
vgg16_tl_weights_path = '/Users/Lavector/testdata/weights_0015.hdf5'
# xml_path = '/users/lavector/testdata/vgg_tl_feature_025.xml'
xml_path = '/Users/Lavector/testdata/vgg_tl_feature_15_caltech101.xml'

eval_save_path = '/Users/Lavector/testdata/result_0222'

# dist_zip, _ = search(anchor_path, search_path, 'vgg_16_tl', vgg16_tl_weights_path, save_path = xml_path)

# dist_zip,_ = search(anchor_path, search_path, 'vgg_16', vgg16_weights_path, save_path=xml_path)

# acc, X, Y = evaluation(eval_path, 'vgg_16_tl', vgg16_tl_weights_path, xml_path, top_n = 5, eval_save_path = eval_save_path)
#print acc
#outs = get_feature(search_path, 'vgg_16', vgg16_weights_path, xml_path)


im1 = read_img('/users/lavector/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg', (224,224))
