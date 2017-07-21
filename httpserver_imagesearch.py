#coding=utf-8

import json
from flask import Flask, request
from collections import OrderedDict
from elasticsearch import Elasticsearch
from image_search.elasticsearch_driver import SignatureES
import socket

hostname = socket.gethostname()

app = Flask(__name__)

port = '3005'
es_index = "image_search0"
# weight_path='/share/image_search_model/vgg16_weights.h5'
weight_path='/Users/Lavector/model/vgg16_weights.h5'

# es = Elasticsearch("es1")
es = Elasticsearch("0.0.0.0", port=9210, timeout=15)

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
	try:
		# create index
		es.indices.create(index=es_index, body=settings)
	except Exception, e:
		pass

ses = SignatureES(es, index=es_index, weight_path=weight_path)

@app.route('/search',methods=['GET', 'POST'])
@app.route('/search/<int:term_after>',methods=['GET', 'POST'])
def message_for_image_search(term_after=5):
	if request.method == 'GET':
		return '200'
	print term_after
	if term_after <=0 or term_after > 100:
		return '201'
	search_dict = {}
	search_dict["tag"] = {}
	search_dict["result"] = "ok"
	search_dict["error_message"] = "no error"
	re = request.form['files']
	print re

	try:
		result = json.loads(re)
		print result
	except :
		search_dict["result"] = "error"
		search_dict["error_message"] = "can not catch picture"

	else:
		message_search = OrderedDict()
		if 'pics' in result.keys() and (result['pics'] is not None) and (len(result['pics']) > 0):
			for each_pic in result['pics']:
				url = each_pic['url']
				pic_id = each_pic['id']
				print url

				try:
					result = ses.search_image(str(url), term_after)
					message_search[pic_id] = result
				except Exception, e:
					message_search[pic_id] = []
					search_dict["result"] = "error"
					search_dict["error_message"] = str(e)


		search_dict['tag'] = message_search
	# print "detection_dict: ", detection_dict
	tag_json = json.dumps(search_dict)
	print tag_json
	return tag_json

@app.route('/add',methods=['GET', 'POST'])
def message_for_image_add():
	if request.method == 'GET':
		return '200'
	search_dict = {}
	search_dict["result"] = "ok"
	search_dict["error_message"] = "no error"
	re = request.form['files']

	try:
		result = json.loads(re)
		msg_id = result["id"]
		print result
	except :
		search_dict["result"] = "error"
		search_dict["error_message"] = "can not catch picture"

	else:
		if 'pics' in result.keys() and (result['pics'] is not None) and (len(result['pics']) > 0):
			for each_pic in result['pics']:
				url = each_pic['url']
				pic_id = each_pic['id']
				print url

				try:
					ses.add_image(str(url), str(msg_id), str(pic_id))
				except Exception, e:
					search_dict["result"] = "error"
					search_dict["error_message"] = str(e)

	# print "detection_dict: ", detection_dict
	tag_json = json.dumps(search_dict)
	print tag_json
	return tag_json

if __name__ == '__main__':
	app.run(debug=False, host='0.0.0.0', port=port)
