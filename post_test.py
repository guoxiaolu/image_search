#coding=utf-8
import requests
import json
import time


message_search = { "id" : "weibo_3992350184721547",
            "site" : 'dianping11111',
            "pics" : [ {"url":  "https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/u=3899668330,1902168048&fm=11&gp=0.jpg", "id" : "b9c7afaf2edda3cc5011dfcd07e93901233f9253"}]
}

message_add = { "id" : "weibo_3992350184721547",
            "site" : 'dianping11111',
            "pics" : [ {"url": "https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/u=3899668330,1902168048&fm=11&gp=0.jpg", "id" : "b9c7afaf2edda3cc5011dfcd07e93901233f9254"}]
}



temp =  json.dumps(message_search)
payloadfiles = {'files':temp}

start = time.time()
r = requests.post("http://0.0.0.0:3005/search",data=payloadfiles)
print time.time() - start
print r.text
# for i in range(50):
#     start = time.time()
#     r = requests.post("http://0.0.0.0:3005/search",data=payloadfiles)
#     print i, time.time() - start
    # print r.text

# temp =  json.dumps(message_add)
# payloadfiles = {'files':temp}
# start = time.time()
# r = requests.post("http://0.0.0.0:3005/add",data=payloadfiles)
# print time.time() - start
# print r.text



# import json
#
# total = 0.0
# num = 0
# f = open('../es_image_search.json', 'r')
# for line in f.readlines():
#     num += 1
#     if num > 1000:
#         break
#     # if num < 94443:
#     #     continue
#     data = json.loads(line)['_source']
#     message = {}
#     message['id'] = data['msg_id']
#     message_pic = {}
#     message_pic['url'] = data['path']
#     message_pic['id'] = data['pic_id']
#     message['pics'] = [message_pic]
#
#     payloadfiles = {'files': json.dumps(message)}
#     start = time.time()
#     r = requests.post("http://0.0.0.0:3005/search", data=payloadfiles)
#     cost = time.time() - start
#     print cost
#     print num, message_pic['url'], r.text
#     total += cost
#
# f.close()
# print 'cost:%f'%(total/num)

# num = 0
# f = open('../es_image_search.json','r')
# for line in open('../es_image_search.json','r'):
#     line = f.readline()
#     print line
#     num += 1
# print num
