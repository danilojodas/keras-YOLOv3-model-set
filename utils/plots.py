# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 22:56:27 2020

@author: DANILO
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def loss_function(folder='', save_plot=False):
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)
    plt.rc('legend', fontsize=11)
    plt.rc('axes', titlesize=14)
    
    fig = plt.figure(figsize=(9,4))
    
    for i in range(2):
        path_ = 'D:/Projetos Machine Learning/Python/YOLOv3-keras/results/v' + str(i+1)
        loss_train = pd.read_csv(os.path.join(path_,'train-epoch_loss.csv'),delimiter=',')
        loss_valid = pd.read_csv(os.path.join(path_,'validation-epoch_loss.csv'),delimiter=',')
        
        ax = fig.add_subplot(1,2,i+1)
        
        ax.plot(np.arange(1,len(loss_train)+1), loss_train.iloc[:,-1],label='Train',antialiased=True)
        ax.plot(np.arange(1,len(loss_valid)+1), loss_valid.iloc[:,-1],label='Validation',antialiased=True)
        
        ax.set_xlabel('Epoch\n\n')
        ax.set_ylabel('Loss')
        ax.set_title('Setup ' + str(i+1))    
        ax.set_ylim([0,100])
    
    plt.legend()
    fig.tight_layout(pad=3)
    
    if (save_plot):
        plt.savefig('D:/Vida académica/Pós-doc/Publicações/Tree Detection/tex - Springer/Figures/loss_cnn.eps',format='eps',
                    dpi=600,bbox_inches='tight')
        
def bar_char():
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)
    plt.rc('legend', fontsize=11)
    plt.rc('axes', titlesize=14)
    
    labels = ('stick','tree','crown','stem')
    x_map = np.array([0.94,0.78,0.72,0.68])
    y_pos = np.arange(len(labels))
    
    fig, ax = plt.subplots()
    
    ax.barh(y_pos,x_map)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Average precision')
    ax.set_title('Mean Average Precision')