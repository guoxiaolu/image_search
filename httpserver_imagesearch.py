#coding=utf-8

import json
from flask import Flask, request
from collections import OrderedDict
from elasticsearch import Elasticsearch
from image_search.elasticsearch_driver import SignatureES
import socket

hostname = socket.gethostname()

app = Flask(__name__)

port = '9201'
imgserver_ip = hostname
imgserver_port = '9202'

thumbnail_path = '../thumbnail'

es = Elasticsearch("es1")
ses = SignatureES(es, index='image_search', weight_path='/share/image_search_model/vgg16_weights.h5', save_path=thumbnail_path, imgserver_ip = imgserver_ip, imgserver_port = imgserver_port)

# es = Elasticsearch("0.0.0.0")
# ses = SignatureES(es, index='image_search', weight_path='/Users/Lavector/model/vgg16_weights.h5', save_path=thumbnail_path, imgserver_ip = imgserver_ip, imgserver_port = imgserver_port)

@app.route('/search',methods=['GET', 'POST'])
def message_for_image_search():
	if request.method == 'GET':
		return '200'
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
					result = ses.search_image(str(url))
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
	app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
