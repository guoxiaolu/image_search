# image_search
image search based on elastic search, image-match and deep learning

Algorithm: Use VGG_16 model to extract global feature of image, deep learning model is implemented by Keras. (theano backend) The distance metric is based on cosine_distance.
Search Engine: Elastic search

This project is modified from [image-match](https://github.com/ascribe/image-match).
The VGG_16 model can be get from [this](http://files.heuritech.com/weights/vgg16_weights.h5).

REST has been added to this project based on flask, gunicorn is also supported. 
Dependence:
  basic package: numpy, scipy, skimage
  keras/theano
  elasticsearch
  flask

