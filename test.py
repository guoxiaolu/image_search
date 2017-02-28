from elasticsearch import Elasticsearch
from image_search.elasticsearch_driver import SignatureES
import matplotlib.pyplot as plt
from skimage.io import imread


"""Given two images, output the distance"""
# from image_search.image_signature import ImageSignature
# gis = ImageSignature()
# a = gis.generate_signature('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
# b = gis.generate_signature('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
# print gis.normalized_distance(a, b)


es = Elasticsearch()
ses = SignatureES(es, distance_cutoff=0.095)

"""Add images to elastic search,"""
# import os.path
# import glob
# from tqdm import tqdm
# import re
# R = re.compile('(\.jpg|\.jpeg|\.bmp|\.png)$')
# add image to image set
# img_path = '/users/lavector/testdata/test3'
# all_files = glob.glob(os.path.join(img_path, '*.*'))
# for fn in tqdm(all_files):
#     if R.search(fn) != None:
#         ses.add_image(fn)

# ses.delete_duplicates('/users/lavector/testdata/test3/images18.jpg')

"""Display search result"""
search_path = 'https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/u=61419631,1940012873&fm=21&gp=0.jpg'
# search_path = '/users/lavector/testdata/testset/accordion-image_0005.jpg'

result = ses.search_image(search_path)
print len(result)

plt_result = plt.subplot(331)
plt_result.imshow(imread(search_path))
plt.axis('off')
for i, re in enumerate(result[:min(6, len(result))]):
    number = int('33%d'%((i+4)))
    plt_result = plt.subplot(number)
    path = re['path']
    print path, re['dist'][0]
    plt_result.imshow(imread(path))
    plt.axis('off')
plt.show()


"""some useful instruction about Elasticsearch"""
# es.search(index='images', _source_include = ['path'])
# es.count()
# es.delete()