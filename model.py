'''
title: recurrent mask rcnn
date: 2018/4/10
author: Junior Liu
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# TODO: box+delta; chop regions;nms
class ProposalLayer:
    """Receives anchor scores and selects a subset to pass as proposals
     to the second stage. Filtering is done based on anchor scores and
     non-max suppression to remove overlaps. It also applies bounding
     box refinement deltas to anchors.

     Inputs:
         rpn_probs: [batch, anchors, (bg prob, fg prob)]
         rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

     Returns:
         Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
     """
    pass


# TODO: Pyramid ROI layer
class PyramidROIAlign:
    """
    Implements ROI Pooling on multiple levels of the feature pyramid.
    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels
    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                  [P2, P3, P4, P5]. Each has a different resolution.
                  Each is [batch, height, width, channels]
    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    pass


# TODO: IoU calculation; generate labels(cls_list,gt_cls,gt_region,gt_mask)
class DetectionTargetLayer:
    '''Maybe dont nd to generate labels, Prepare_Data'''
    pass


# TODO: combine the info of 2nd cls and reg to complete final detection
class DetectionLayer:
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.
    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are in image domain
    """
    pass


# TODO: accomplish FPN heads
'''
FPN heads
'''


# TODO: define loss functions
'''
loss function
'''


# TODO:the main frame of Mask R-cnn
'''
Mask R-CNN class
'''



if __name__ == '__main__':
    print('hello world!')