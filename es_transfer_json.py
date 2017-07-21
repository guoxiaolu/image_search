# modify es
from elasticsearch import Elasticsearch
import time
import numpy as np
import json
from image_search.get_label import get_labels, read_labels


def desigmoid(z):
    return -np.log(1.0 / z - 1.0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def signature_process(signature):
    return [float(s.split('|')[1]) for s in signature.split(' ')]

def sig_list_to_delim(sig_list):
    sig_res = ''
    for i, val in enumerate(sig_list):
        if i != (len(sig_list) - 1):
            sig_res += '%d|%f ' % (i, val)
        else:
            sig_res += '%d|%f' % (i, val)
    return sig_res

es = Elasticsearch("0.0.0.0", port=9200)
es_index = 'image_search0'

labels_id, labels_name = read_labels()

print 'start add from json'
num = 0
json_name = '../image_search_test'
f = open(json_name, 'r')
for line in f:
    # line = f.readline()
    num += 1

    data = json.loads(line)['_source']

    signature = data['signature']

    signature_new = sig_list_to_delim(signature)

    # print signature
    # print signature_new
    signature_desigmoid = desigmoid(np.array(signature))
    label = softmax(signature_desigmoid)
    top_3 = np.argsort(label)[::-1][:3]
    top_3_prob = label[top_3]
    label_filter = get_labels(labels_id, labels_name, top_3, top_3_prob)

    message = {}
    message['msg_id'] = data['msg_id']
    message['pic_id'] = data['pic_id']
    message['path'] = data['path']
    message['timestamp'] = data['timestamp']

    message['top_1'] = label_filter[0]
    message['top_2'] = label_filter[1]
    message['top_3'] = label_filter[2]

    message['signature'] = signature_new

    try:
        es.index(index=es_index, doc_type='image', body=message, timeout='20s')
    except Exception, e:
        print num, e
        continue
    print num

f.close()
