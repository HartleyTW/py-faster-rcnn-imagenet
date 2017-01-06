#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

#CLASSES = ['__background__']

with open(os.path.join('output','classesTrained.txt')) as f:
    lines = f.read().splitlines()
    
CLASSES = lines
print CLASSES

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'imNetVGG' : ('VGG16', 
                  'VGG16.v2.caffemodel')}


def vis_detections(image_name, im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    print len(inds)
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s}'.format(class_name),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')

    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    # save the image
    fig = plt.gcf()
    print 'saving'
    fig.savefig("output_"+class_name+"_"+image_name)

def vis_all_in_one(image_name, im, class_dets, thresh=0.5):
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    

    for cla in class_dets:
        for detection in class_dets[cla]:
            bbox = detection[:4]
            score = detection[-1]

            ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )

            ax.text(bbox[0], bbox[1] - 2,
                '{:s}'.format(cla),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
            # ax.text(bbox[0], bbox[1] - 2,
            #     '{:s} {:.3f}'.format(cla, score),
            #     bbox=dict(facecolor='blue', alpha=0.5),
            #     fontsize=14, color='white')

            #ax.set_title(('Detections with threshold >= {:.1f}').format(thresh), fontsize=14)

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    fig = plt.gcf()
    print 'saving combined'
    fig.savefig("frames/output_comb_"+image_name)


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name)
    im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', 'frames', image_name)
    print im_file
    im = cv2.imread(im_file)

    # resize shorter side to 600 px
    #print im.shape
    # if im.shape[0] < im.shape[1]:
    #     ratio = 600.0 / im.shape[0]
    #     dim = (int(im.shape[1]*ratio), 600)
    # else :
    #     ratio = 600.0 / im.shape[1]
    #     dim = (600, int(im.shape[0]*ratio))


    # im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    #print im.shape

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.1#0.7
    NMS_THRESH = 0.3
    all_cls_dets = {}

    for cls_ind, cls in enumerate(CLASSES[1:]):
        print cls
        bboxes = []
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        print len(scores)
        print cls_ind
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #vis_detections(image_name, im, cls, dets, thresh=CONF_THRESH)

        #print np.greater(dets[:,-1],CONF_THRESH)
        if np.any(np.greater(dets[:,-1],CONF_THRESH)):
            for i in dets[np.greater(dets[:,-1],CONF_THRESH),:]:
                bboxes.append(i)
            all_cls_dets[cls] = bboxes

        #print all_cls_dets
    vis_all_in_one(image_name, im, all_cls_dets,CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    prototxt = os.path.join(cfg.ROOT_DIR, 'models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt')

    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])
    caffemodel = 'output/faster_rcnn_end2end/train/vgg16_faster_rcnn_iter_500000.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']

    im_names = ['1.jpg','2.jpg','3.jpg','4.jpg','4.5.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg','10.jpg']#,'1.png','2.png','3.png','3.5.png','4.png','4.5.png','5.png','6.png','orange1.png','orange2.png']
    im_names = os.listdir('data/demo/frames')
    print im_names

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)
    #plt.show()
    print 'done'
