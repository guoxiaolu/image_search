#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:56:34 2017

@author: Lavector
"""

#from xml.dom.minidom import parse
#import xml.dom.minidom
#
## Open XML document using minidom parser
#DOMTree = xml.dom.minidom.parse("/users/lavector/movie.xml")
#collection = DOMTree.documentElement
#if collection.hasAttribute("shelf"):
#   print "Root element : %s" % collection.getAttribute("shelf")
#
## Get all the movies in the collection
#movies = collection.getElementsByTagName("movie")
#
## Print detail of each movie.
#for movie in movies:
#   print "*****Movie*****"
#   if movie.hasAttribute("title"):
#      print "Title: %s" % movie.getAttribute("title")
#
#   type = movie.getElementsByTagName('type')[0]
#   print "Type: %s" % type.childNodes[0].data
#   format = movie.getElementsByTagName('format')[0]
#   print "Format: %s" % format.childNodes[0].data
#   rating = movie.getElementsByTagName('rating')[0]
#   print "Rating: %s" % rating.childNodes[0].data
#   description = movie.getElementsByTagName('description')[0]
#   print "Description: %s" % description.childNodes[0].data
#

 
from xml.dom.minidom import Document
from xml.etree.ElementTree import parse
class XMLGenerator:  
    def __init__(self, xml_name):  
        self.doc = Document()  
        self.xml_name = xml_name
    def createNode(self, node_name):  
        return self.doc.createElement(node_name)  
    def addNode(self, node, prev_node = None):  
        cur_node = node  
        if prev_node is not None:  
            prev_node.appendChild(cur_node)  
        else:  
            self.doc.appendChild(cur_node)  
        return cur_node  
    def setNodeAttr(self, node, att_name, value):  
        cur_node = node  
        cur_node.setAttribute(att_name, value)  
    def setNodeValue(self, cur_node, value):  
        node_data = self.doc.createTextNode(value)  
        cur_node.appendChild(node_data)  
    def genXml(self):  
        f = open(self.xml_name, "w")  
        f.write(self.doc.toprettyxml(indent = "\t", newl = "\n", encoding = "utf-8"))
        f.close()  
        
class XMLParser:
    def __init__(self, xml_name):
        self.root = parse(xml_name).getroot()
    def getAllNodes(self, name):
        return self.root.findall(name)
    def getNodeValue(self, node, name):
        return node.find(name).text
    def getRootAttrib(self, name):
        return self.root.attrib[name]

#doc = XMLParser('/users/lavector/features.xml')
#imgs = doc.getNode('img')
#for img in imgs:
#    node = doc.getNode('label')
#    data = doc.getNodeValue(node)

        
#doc = XMLGenerator('/users/lavector/people.xml')
#root = doc.createNode('img')
#doc.setNodeAttr(root, 'dataset', 'Caltech-101')
#doc.addNode(root)
#
#img = doc.createNode('img')
#doc.addNode(img, root)
#
#idx = doc.createNode('id')
#doc.setNodeValue(idx, '0')
#doc.addNode(idx, img)
#
#path = doc.createNode('path')
#doc.setNodeValue(path, '/users/lavector/img.jpg')
#doc.addNode(path, img)
#
#feature = doc.createNode('feature')
#doc.setNodeValue(feature, '12,13')
#doc.addNode(feature, img)
#
#doc.genXml()           

#img.setAttribute('path', '/users/lavector/img.jpg')
#img.setAttribute('feature', '12,13')
