# image_search
image search based on elastic search, image-match and deep learning

Algorithm: use VGG_16 model to extract global feature of image, deep learning model is implemented by Keras. (theano backend)
Search Engine: Elastic search

This project is modified from [image-match](https://github.com/ascribe/image-match).
The VGG_16 model can be get from [this](http://files.heuritech.com/weights/vgg16_weights.h5).

Dependence:
  basic package: numpy, scipy, skimage
  keras/theano
  elasticsearch
