#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:18:09 2017

@author: Lavector
"""

import numpy as np
from scipy.misc import imread, imresize
import glob
import os
import random

def read_img(fname, img_size = (224, 224, 3)):
    img = imread(fname, mode = 'RGB')
    img = imresize(img, img_size)
    img = img.astype('float32')
    # We normalize the colors (in RGB space) with the empirical means on the training set
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = img.transpose((2, 0, 1))
    return img
    
def gen_triplet(pos, neg, path, aug):
    anchor_list = []
    pos_list = []
    neg_list = []
    l_pos = len(pos)
    l_neg = len(neg)
    for i, f in enumerate(pos):
        fname = os.path.join(path, f + '.jpg')
        img = read_img(fname)
        
        for k in range(aug):
            r = random.randint(0, l_pos - 1)
            while r == i:
                r = random.randint(0, l_pos - 1)
            fname = os.path.join(path, pos[r] + '.jpg')
            img_pos = read_img(fname)
            
            r = random.randint(0, l_neg - 1)
            fname = os.path.join(path, neg[r] + '.jpg')
            img_neg = read_img(fname)
            
            anchor_list.append(img)
            pos_list.append(img_pos)
            neg_list.append(img_neg)
    return anchor_list, pos_list, neg_list
    
#def img_gen(txt_path, img_path, aug = 5, start_r = 0, end_r = 0.8):
#    all_files = glob.glob(os.path.join(txt_path, '*_trainval.txt'))
#    labels = []
#    
#    for txt in all_files:
#        anchor_list = []
#        pos_list = []
#        neg_list = []
#        basename = os.path.basename(txt)
#        labels.append(basename.split('_')[0])
#        f = open(txt)
#        pos = []
#        neg = []
#        for line in f.readlines():
#            content = (line.strip()).split(' ')
#            if content[-1] == '1':
#                pos.append(content[0])
#            else:
#                neg.append(content[0])
#        
#        pos_len = len(pos)
#        neg_len = len(neg)
#        if pos_len * aug > 12000:
#            a, p, n = gen_triplet(pos[int(pos_len*start_r):12000/aug], neg[int(neg_len*start_r):int(neg_len*end_r)], img_path, aug)
#        else:
#            a, p, n = gen_triplet(pos[int(pos_len*start_r):int(pos_len*end_r)], neg[int(neg_len*start_r):int(neg_len*end_r)], img_path, aug)
#        anchor_list += a
#        pos_list += p
#        neg_list += n
#        print basename.split('_')[0], len(anchor_list)
#
#        anchor = np.array(anchor_list)
#        pos = np.array(pos_list)
#        neg = np.array(neg_list)
#        yield anchor, pos, neg

def gen_tripfile(pos, neg, aug):
    l_pos = len(pos)
    l_neg = len(neg)
    for i, f in enumerate(pos):
        for k in range(aug):
            r = random.randint(0, l_pos - 1)
            while r == i:
                r = random.randint(0, l_pos - 1)            
            n = random.randint(0, l_neg - 1)
            
            yield f, pos[r], neg[n]
    
def gen_filelist(txt_path, img_path, save_path, aug = 1):
    all_files = glob.glob(os.path.join(txt_path, '*_trainval.txt'))
    labels = []
    
    wf = open(save_path, 'w')
    for txt in all_files[::3]:
        basename = os.path.basename(txt)
        labels.append(basename.split('_')[0])
        f = open(txt)
        pos = []
        neg = []
        for line in f.readlines():
            content = (line.strip()).split(' ')
            if content[-1] == '1':
                pos.append(content[0])
            else:
                neg.append(content[0])
        
        for a, p, n in gen_tripfile(pos, neg, aug):
            content = '%s;%s;%s\n'%(a, p, n)
            wf.writelines(content)
        
        f.close()
    wf.close()

def img_gen(filename, img_path, batch_size = 1000):
    f = open(filename)
    anchor_list = []
    pos_list = []
    neg_list = []
    for line in f.readlines():
        content = (line.strip()).split(';')
        if len(content) != 3:
            continue
        fname = os.path.join(img_path, content[0] + '.jpg')
        img = read_img(fname)
        fname = os.path.join(img_path, content[1] + '.jpg')
        img_pos = read_img(fname)
        fname = os.path.join(img_path, content[2] + '.jpg')
        img_neg = read_img(fname)
        
        anchor_list.append(img)
        pos_list.append(img_pos)
        neg_list.append(img_neg)
        
        if len(anchor_list) == batch_size:
            anchor = np.array(anchor_list)
            pos = np.array(pos_list)
            neg = np.array(neg_list)
            yield anchor, pos, neg
            anchor_list = []
            pos_list = []
            neg_list = []
    f.close()    
    if len(anchor_list) > 0:
        anchor = np.array(anchor_list)
        pos = np.array(pos_list)
        neg = np.array(neg_list)
        yield anchor, pos, neg
        
def img_all(filename, img_path):
    f = open(filename)
    anchor_list = []
    pos_list = []
    neg_list = []
    for line in f.readlines():
        content = (line.strip()).split(';')
        if len(content) != 3:
            continue
        fname = os.path.join(img_path, content[0] + '.jpg')
        img = read_img(fname)
        fname = os.path.join(img_path, content[1] + '.jpg')
        img_pos = read_img(fname)
        fname = os.path.join(img_path, content[2] + '.jpg')
        img_neg = read_img(fname)
        
        anchor_list.append(img)
        pos_list.append(img_pos)
        neg_list.append(img_neg)
        
    f.close()    
    anchor = np.array(anchor_list)
    pos = np.array(pos_list)
    neg = np.array(neg_list)
    return anchor, pos, neg
        

#txt_path = '/Users/Lavector/dataset/VOC2012/ImageSets/label'
#img_path = '/Users/Lavector/dataset/VOC2012/JPEGImages'
#save_path = '/Users/Lavector/val.txt'

#anchor, pos, neg = get_img(txt_path, img_path)

#for anchor, pos, neg in img_gen(save_path, img_path):
#    print len(anchor)

#gen_filelist(txt_path, img_path, save_path)

#anchor, pos, neg = img_all('/Users/Lavector/test1.txt', img_path)