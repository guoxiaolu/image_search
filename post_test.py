#coding=utf-8
import requests
import json


message_search = { "id" : "weibo_3992350184721547",
            "site" : 'dianping11111',
            # "pics" : [ {"url": "https://ss1.bdstatic.com/70cFvXSh_Q1YnxGkpoWK1HF6hhy/it/u=2338765460,3992891691&fm=23&gp=0.jpg", "id" : "b9c7afaf2edda3cc5011dfcd07e93901233f9253"},{"url": "http://img1.gamersky.com/image2016/08/20160803_my_227_3/1.jpg", "id": "005GOo6qgw1fattbgyyrrkj30qr0qozpf"}]
            "pics" : [ {"url": "https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/u=3899668330,1902168048&fm=11&gp=0.jpg", "id" : "b9c7afaf2edda3cc5011dfcd07e93901233f9253"},{"url": "http://img1.gamersky.com/image2016/08/20160803_my_227_3/1.jpg", "id": "005GOo6qgw1fattbgyyrrkj30qr0qozpf"}]
}

message_add = { "id" : "weibo_3992350184721547",
            "site" : 'dianping11111',
            "pics" : [ {"url": "https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/u=3899668330,1902168048&fm=11&gp=0.jpg", "id" : "b9c7afaf2edda3cc5011dfcd07e93901233f9254"}]
}



temp =  json.dumps(message_search)
payloadfiles = {'files':temp}
r = requests.post("http://0.0.0.0:9201/search",data=payloadfiles)
print r.text

# temp =  json.dumps(message_add)
# payloadfiles = {'files':temp}
# r = requests.post("http://0.0.0.0:9201/add",data=payloadfiles)
# print r.text

