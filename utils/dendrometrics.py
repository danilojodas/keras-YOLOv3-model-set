# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:42:06 2020

@author: DANILO
"""

import os
import math
import pandas as pd
import numpy as np
from collections import defaultdict

def save_dendrometric(image_folder, data_file, out_file='dendrometrics.txt'):
    #image_folder = 'D:/Projetos Machine Learning/Python/YOLOv3-keras/dataset/validation/image'    
    #data_file = 'D:/Vida académica/Pós-doc/Projeto/Computador do lab/Dataset/IPT/dataset_new.csv'
    
    image_list = os.listdir(image_folder)
    
    df = pd.read_csv(data_file, delimiter=';', engine='python',encoding='utf_8_sig')
    
    counter = 0
    species_dict = defaultdict(list)
    metrics = list()
    
    for img in image_list:
        img_cod = img.split('.')[0]
        if (img_cod in list(df.iloc[:,0])):
            counter+=1
            specie = df.loc[df['Código Árvore'] == img_cod]['Espécie'].iloc[0]
            height = df.loc[df['Código Árvore'] == img_cod]['Altura da Árvore (m)'].iloc[0]
            dbh = df.loc[df['Código Árvore'] == img_cod]['Diâmetro Altura do Peito (m)'].iloc[0]
            dc = df.loc[df['Código Árvore'] == img_cod]['Diâmetro da Copa (m)'].iloc[0]
            metrics.append([img, height, dbh, dc, specie])
            species_dict[specie].append(specie)
    
    metrics = np.vstack(metrics)
    
    np.savetxt(out_file, metrics, delimiter=',', fmt='%s',
               header='Codigo Arvore, Altura, DAP, Diametro Copa, Especie')

def load_dendrometric(file_name):
    try:
        data = np.loadtxt(file_name, dtype=str, delimiter=',')
        return data
    except:
        return None

def calculate_dendrometrics(class_names, out_boxes, out_classes):
    idx_tree = class_names.index('tree')
    idx_stick = class_names.index('stick')
    idx_crown = class_names.index('crown')
    
    tree, = np.where(out_classes == idx_tree)
    stick, = np.where(out_classes == idx_stick)
    crown, = np.where(out_classes == idx_crown)
    
    tree_height_m = 0
    diameter_crown = 0
    
    if (len(tree) > 0 and len(stick) > 0):
        stick_height_m = 3
        stick_height_px = out_boxes[stick[0],-1] - out_boxes[stick[0],1]        
        
        if (len(tree) > 0):                        
            tree_height_px = out_boxes[tree[0],-1] - out_boxes[tree[0],1]            
            tree_height_m = (tree_height_px * stick_height_m) / stick_height_px
        
        if (len(crown) > 0):
            crown_height_px = out_boxes[crown[0],-1] - out_boxes[crown[0],1]
            crown_width_px = out_boxes[crown[0],2] - out_boxes[crown[0],0]
            
            diameter_crown = math.sqrt(pow(crown_width_px / stick_height_px * stick_height_m,2) + pow(crown_height_px / stick_height_px * stick_height_m,2))
    
    return tree_height_m, diameter_crown