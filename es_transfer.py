# modify es
from elasticsearch import Elasticsearch
import time
import numpy as np
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
es1 = Elasticsearch("0.0.0.0", port=9210)
es_index = 'image_search'
if not es.indices.exists(es_index):
	# index settings
	settings = {
		"settings" : {
			"analysis": {
					"analyzer": {
					   "payload_analyzer": {
						  "type": "custom",
						  "tokenizer":"whitespace",
						  "filter":"delimited_payload_filter"
						}
			  }
			}
		 },
		 "mappings": {
			"image" : {
			  "properties" : {
				  "signature": {
							  "type": "text",
							  "term_vector": "with_positions_offsets_payloads",
							  "analyzer" : "payload_analyzer"
						   }
			  }
		  }
		 }
	}
	# create index
	es.indices.create(index=es_index, body=settings)

start = time.time()
page = es1.search(
    index='image_search',
    doc_type='image',
    scroll='1m',
    size=1000)
end_scroll = time.time()
print end_scroll-start

sid = page['_scroll_id']
scroll_size = page['hits']['total']

labels_id, labels_name = read_labels()
num = 0
print 'start scrolling'
# Start scrolling
while (scroll_size > 0):
    start = time.time()
    # Get the number of results that we returned in the last scroll
    res = page['hits']['hits']
    scroll_size = len(res)
    num += scroll_size
    print num
    for r in res:
        r_new = {}
        r_s = r['_source']
        signature = r_s['signature']
        signature_new = sig_list_to_delim(signature)
        signature_desigmoid = desigmoid(np.array(signature))
        label = softmax(signature_desigmoid)
        top_3 = np.argsort(label)[::-1][:3]
        top_3_prob = label[top_3]
        label_filter = get_labels(labels_id, labels_name, top_3, top_3_prob)
        r_new['top_1'] = label_filter[0]
        r_new['top_2'] = label_filter[1]
        r_new['top_3'] = label_filter[2]
        r_new['timestamp'] = r_s['timestamp']
        r_new['msg_id'] = r_s['msg_id']
        r_new['pic_id'] = r_s['pic_id']
        r_new['path'] = r_s['path']
        r_new['signature'] = signature_new
        es.index(index=es_index, doc_type='image', body=r_new)
    page = es1.scroll(scroll_id=sid, scroll='1m')
    # Update the scroll ID
    sid = page['_scroll_id']
    end_scroll = time.time()
    print end_scroll - start
