# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 09:26:05 2020

@author: DANILO
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def crown_segmentation(img_file_name, out_boxes, out_classes, class_names):
    idx_crown = class_names.index('crown')
    
    crown_idx, = np.where(out_classes == idx_crown)
    
    if (len(crown_idx) > 0):
        #-----------------------------------
        ## 1. Create Chromaticity Vectors ##
        #-----------------------------------
        
        # Get Image
        img = cv2.imread(img_file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                
        
        # Get the area corresponding to the crown
        idx_ = crown_idx[0]
        img = img[out_boxes[idx_][1]:out_boxes[idx_][-1],out_boxes[idx_][0]:out_boxes[idx_][2],:]
        
        h, w = img.shape[:2]    
        
        img = cv2.GaussianBlur(img, (5,5), 0)
        
        # Separate Channels
        r, g, b = cv2.split(img) 
        
        im_sum = np.sum(img, axis=2)
        
        rg_chrom_r = np.ma.divide(1.*r, im_sum)
        rg_chrom_g = np.ma.divide(1.*g, im_sum)
        rg_chrom_b = np.ma.divide(1.*b, im_sum)
        
        # Visualize rg Chromaticity --> DEBUGGING
        rg_chrom = np.zeros_like(img)
        
        rg_chrom[:,:,0] = np.clip(np.uint8(rg_chrom_r*255), 0, 255)
        rg_chrom[:,:,1] = np.clip(np.uint8(rg_chrom_g*255), 0, 255)
        rg_chrom[:,:,2] = np.clip(np.uint8(rg_chrom_b*255), 0, 255)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        rg_chrom_g_int = np.array(rg_chrom_g*255, dtype=np.uint8)
        
        _, imgf = cv2.threshold(hsv[:,:,2], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, imgf_chrom = cv2.threshold(rg_chrom_g_int, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
        closing = cv2.morphologyEx(imgf_chrom, cv2.MORPH_CLOSE, kernel)
        mask_chrom = closing > 0
        
        crown = np.zeros_like(img, np.uint8)
        crown[mask_chrom] = img[mask_chrom]
        
        return crown
    else:
        return None