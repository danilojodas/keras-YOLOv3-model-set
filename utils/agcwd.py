# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 13:53:16 2020

@author: DANILO
"""

import numpy as np

def pdf(arr):      
    min_value = np.min(arr)
    max_value = np.max(arr)
    
    pdf_distr = np.zeros(max_value+1)    
    
    for g in range(min_value,max_value):
        pdf_distr[g] = len(arr[arr == g])
    
    return pdf_distr

def pdf_w(pdf_distr,alpha=0.3):
    distr_len = len(pdf_distr)
    
    pdf_w_distr = np.zeros(distr_len)
    
    pdf_min = np.min(pdf_distr)
    pdf_max = np.max(pdf_distr)
    
    for g in range(distr_len):
        pdf_w_distr[g] = pdf_max*np.power(((pdf_distr[g] - pdf_min) / (pdf_max - pdf_min)),alpha)
    
    return pdf_w_distr

def cdf_w(pdf_w_distr):
    distr_len = len(pdf_w_distr)
    
    cdf_w_distr = np.zeros(distr_len)
    
    sum_pdf = np.sum(pdf_w_distr)
    
    for g in range(distr_len):
        cdf_w_distr[g] = np.sum(pdf_w_distr[0:g+1]) / sum_pdf
    
    return cdf_w_distr

def contrast_correction(img):
    arr = img.flatten()
    
    new_img = np.zeros((img.shape[0],img.shape[1]))
    
    pdf_distr = pdf(arr)
    pdf_w_distr = pdf_w(pdf_distr,alpha=1)
    
    cdf_w_distr = cdf_w(pdf_w_distr)    
    l_max = np.max(arr)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i][j] = l_max * np.power((img[i][j] / l_max),1-cdf_w_distr[img[i][j]])
            
    return new_img.astype(np.uint8)


    