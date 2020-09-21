# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:43:05 2020

@author: DANILO
"""

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def get_correlation(dendro_metrics, out_file=None):
    pearson_corr = stats.pearsonr(dendro_metrics[:,1].astype(float), dendro_metrics[:,2].astype(float))
    spearman_corr = stats.spearmanr(dendro_metrics[:,1].astype(float), dendro_metrics[:,2].astype(float))
    
    mean_abs_error = [np.mean(abs(dendro_metrics[:,1].astype(float) - dendro_metrics[:,2].astype(float))),
                      np.std(abs(dendro_metrics[:,1].astype(float) - dendro_metrics[:,2].astype(float)))]
    
    plt.figure()
    plt.scatter(dendro_metrics[:,1].astype(float), dendro_metrics[:,2].astype(float))
    plt.xlabel('Automatic (in meters)')
    plt.ylabel('Manual (in meters)')
#    # Pearson
    plt.annotate('{0}'.format('Pearson:'),(15,25),size=14)
    plt.annotate('{:.4f}'.format(pearson_corr[0]),(22,25),size=14)
#    
#    # Spearman
    plt.annotate('{0}'.format('Spearman: '),(15,23),size=14)
    plt.annotate('{:.4f}'.format(spearman_corr[0]),(22,23),size=14)
    plt.title('Height of tree')
    
    if (not out_file is None):
        plt.savefig(out_file,dpi=300,bbox_inches='tight')
    
    return pearson_corr, spearman_corr, mean_abs_error

dendro_metrics = np.loadtxt(os.path.dirname(os.getcwd()) + '/example/with constraints/dendrometric_features.txt',
                            delimiter=',', dtype=object)

idx_nan, _ = np.where(np.isnan(dendro_metrics[:,1:].astype(float)))
dendro_metrics = np.delete(dendro_metrics, idx_nan, axis=0)

p, s, mae = get_correlation(dendro_metrics, out_file=os.path.dirname(os.getcwd()) + '/example/with constraints/correlation_all.pdf')

idx_non_zero, = np.where(dendro_metrics[:,1].astype(float) == 0)
dendro_metrics = np.delete(dendro_metrics, idx_non_zero, axis=0)

p1, s1, mae1 = get_correlation(dendro_metrics, out_file=os.path.dirname(os.getcwd()) + '/example/with constraints/correlation_non_zero.pdf')