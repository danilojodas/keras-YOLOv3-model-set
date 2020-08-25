# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:30:18 2020

@author: DANILO
"""

import numpy as np

def apply_constraints(class_names, out_boxes, out_classes, out_scores):
    print('Applying constraints...')
    idx_stem = class_names.index('stem')
    idx_stick = class_names.index('stick')
    idx_tree = class_names.index('tree')
    idx_crown = class_names.index('crown')
    
    # Stem constraint
    if (idx_stick in out_classes):            
        stick, = np.where(out_classes == idx_stick)
        stems, = np.where(out_classes == idx_stem)

        # Getting the stick with the maximum score
        idx_ = stick[np.argmax(out_scores[stick])]                
        
        # Checking the correct stem
        if (len(stems) > 0):
            higher_weighted_dist = -9999
            higher_weighted_dist_idx = -1
            
            stems_to_remove = list()
            print('index stick: ', str(idx_))
            print('indices stem: ', stems)
            for i in stems:
                # Calculating the weighted difference between the stick and the current stem
                bottom_dist = abs(out_boxes[i][-1] - out_boxes[idx_][-1])
                
                dist = min([abs(out_boxes[idx_][0] - out_boxes[i][2]),
                            abs(out_boxes[idx_][2] - out_boxes[i][0])])
                print('bottom_dist: ', str(bottom_dist))
                print('dist: ', str(dist))
                print('out_scores: ', str(out_scores[i]))
                print('box stick: ', out_boxes[idx_])
                print('box stem ', out_boxes[i])
                weighted_dist = (1 / (bottom_dist + 0.00001)) * (1 / (dist + 0.00001)) * out_scores[i]
                
                if (weighted_dist > higher_weighted_dist):
                    higher_weighted_dist = weighted_dist
                    
                    if (higher_weighted_dist_idx >= 0):
                        stems_to_remove.append(higher_weighted_dist_idx)
                    
                    higher_weighted_dist_idx = i
                else:
                    stems_to_remove.append(i)
            
            out_boxes = np.delete(out_boxes,stems_to_remove,axis=0)
            out_classes = np.delete(out_classes,stems_to_remove,axis=0)
            out_scores = np.delete(out_scores,stems_to_remove,axis=0)
            
            trees, = np.where(out_classes == idx_tree)
            higher_weighted_dist_idx, = np.where(out_classes == idx_stem)
            
            # Checking the correct tree that surronds the stem
            if (len(trees) > 1 and higher_weighted_dist_idx >= 0):
                out_boxes, out_classes, out_scores = remove_duplicate_boxes(trees,
                                                                            out_boxes[higher_weighted_dist_idx,:],
                                                                            out_boxes,
                                                                            out_classes,
                                                                            out_scores)
            crowns, = np.where(out_classes == idx_crown)
            tree_bbox_idx, = np.where(out_classes == idx_tree)
            if (len(crowns) > 1 and len(tree_bbox_idx) > 0):
                out_boxes, out_classes, out_scores = remove_duplicate_boxes(crowns,
                                                                            out_boxes[tree_bbox_idx,:],
                                                                            out_boxes,
                                                                            out_classes,
                                                                            out_scores,
                                                                            method='iou')
    return out_boxes, out_classes, out_scores
        
def overlap_rate(bbox1, bbox2):
    width = bbox1[2] - bbox1[0]
    height = bbox1[-1] - bbox1[1]
    
    condition = list()
    
    for x in range(bbox2[0], bbox2[0] + (bbox2[2] - bbox2[0])):
        for y in range(bbox2[1], bbox2[1] + (bbox2[-1] - bbox2[1])):
            condition.append(bbox1[0] <= x < (bbox1[0]+ width) and bbox1[1] <= y < (bbox1[1] + height))
    
    n = (bbox2[2] - bbox2[0]) * (bbox2[-1] - bbox2[1])
    condition = np.array(condition)
    
    return sum(condition) / n

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou    

def remove_duplicate_boxes(bbox1_ind, bbox2, out_boxes, out_classes, out_scores, method='overlap'):
    if (method!='overlap' and method!='iou'):
        raise Exception('Remove duplicate boxes: method type must be \'overlap\' or \'iou\'!')
    
    larger_overlap = -9999
    larger_overlap_index = -1
    
    indices_to_remove = list()
    
    if (method=='overlap'):
        for i in bbox1_ind:
            # Calculating the overlapping between the ith element of the bbox1 and the bbox2
            overlap = overlap_rate(out_boxes[i], bbox2[0])
            
            if (overlap > larger_overlap):
                larger_overlap = overlap
                
                if (larger_overlap_index >= 0):
                    indices_to_remove.append(larger_overlap_index)
                
                larger_overlap_index = i
            else:
                indices_to_remove.append(i)
    else:
        for i in bbox1_ind:
            # Calculating the overlapping between the ith element of the bbox1 and the bbox2
            overlap = bb_intersection_over_union(out_boxes[i], bbox2[0])
            
            if (overlap > larger_overlap):
                larger_overlap = overlap
                
                if (larger_overlap_index >= 0):
                    indices_to_remove.append(larger_overlap_index)
                
                larger_overlap_index = i
            else:
                indices_to_remove.append(i)            
    
    out_boxes = np.delete(out_boxes,indices_to_remove,axis=0)
    out_classes = np.delete(out_classes,indices_to_remove,axis=0)
    out_scores = np.delete(out_scores,indices_to_remove,axis=0)        
    
    return out_boxes, out_classes, out_scores        