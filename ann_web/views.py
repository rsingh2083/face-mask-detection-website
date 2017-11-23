import os
import json
import base64

from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.template import RequestContext

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def index(request):
	if request.method == 'POST' and request.FILES['img']:
		img = request.FILES['img'].name
		img_idx = img.split('.')[0]
		gt_bboxes = get_gt_bboxes(img_idx)
		pred_bboxes = predict_bboxes(img_idx)
		base64str = encode_base64(img_idx)
#		print('pred_bboxes:', pred_bboxes)# = predict_bboxes(img_idx)
		bboxes = []
		bboxes.append(gt_bboxes)
		bboxes.append(pred_bboxes)
		bboxes.append(base64str)
		print'bboxes:', bboxes# = [gt_bboxes, pred_bboxes]
		return render_to_response('index.html', {
			'bboxes': json.dumps(bboxes), 
			})
	else:
		return render_to_response('index.html')

def get_gt_bboxes(img_id):
	xml_root = '/home/lvyue/py-faster-rcnn/data/VOCdevkit2007/VOC2007/Annotations/'
	xml_file = xml_root + img_id+'.xml'
	if not os.path.isfile(xml_file):
		return []
	root = ET.ElementTree(file=xml_file)
	xmin = []
	ymin = []
	xmax = []
	ymax = []
	for elem in root.iter('object'):
		xmin.append(elem.findall('bndbox')[0].findall('xmin')[0].text)
		ymin.append(elem.findall('bndbox')[0].findall('ymin')[0].text)
		xmax.append(elem.findall('bndbox')[0].findall('xmax')[0].text)
		ymax.append(elem.findall('bndbox')[0].findall('ymax')[0].text)
		
	
	bboxes = []
	for i in range(len(xmin)):
		curr_box = [xmin[i], ymin[i], xmax[i], ymax[i]]
		bboxes.append(curr_box)
		
	print('gt bboxes:', bboxes)
	return bboxes
	

def predict_bboxes(img_idx):
	img_root = '/home/lvyue/py-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages/'
	img_file = img_root + img_idx+'.jpg'
	exec_file = '/home/lvyue/py-faster-rcnn/tools/test.py'
	cmd = 'CUDA_VISIBLE_DEVICES=1 python {} --img {}'.format(exec_file, img_file)
	print(cmd)#= 'python {} --img {}'.format(exec_file, img_file)
	os.system(cmd)
	result_root = '/home/lvyue/py-faster-rcnn/data/test_online/'
	bboxes = []
	with open(result_root + img_idx+'.txt') as f:
		for line in f:
			bbox = line.strip().split(' ')
			bboxes.append(bbox)

	print 'predicted bboxes:', bboxes
	return bboxes


def encode_base64(img_idx):
	img_root = '/home/lvyue/py-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages/'
	img_file = img_root + img_idx+'.jpg'
	with open(img_file) as f:
		str = base64.b64encode(f.read())
	return str
