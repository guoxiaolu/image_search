# image_search
image search based on elastic search, image-match and deep learning

Algorithm: Use VGG_16 model to extract global feature of image, deep learning model is implemented by Keras. (theano backend) The distance metric is based on cosine_distance.
Search Engine: Elastic search

This project is modified from [image-match](https://github.com/ascribe/image-match).
The VGG_16 model can be get from [this](http://files.heuritech.com/weights/vgg16_weights.h5).

About backup: backup is experiment code for so-called Siamese network and triplet-loss function on VOC2012 dataset. The experiment result is worse than the basic feature extraction. If you're intersted in the so-called state-of-art algorithm, try it~ However, instance search that is image search + object detection (Fast rcnn/SSD) is another hot topic.
Dependencies:
  basic package: numpy, scipy, skimage
  keras/theano
  elasticsearch
