# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:43:05 2020

@author: DANILO
"""

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def get_correlation(dendro_metrics, plot_title='Plot', out_file=None):
    pearson_corr = stats.pearsonr(dendro_metrics[:,0].astype(float), dendro_metrics[:,1].astype(float))
    spearman_corr = stats.spearmanr(dendro_metrics[:,0].astype(float), dendro_metrics[:,1].astype(float))
    
    mean_abs_error = [np.mean(abs(dendro_metrics[:,0].astype(float) - dendro_metrics[:,1].astype(float))),
                      np.std(abs(dendro_metrics[:,0].astype(float) - dendro_metrics[:,1].astype(float)))]
    
    plt.figure()
    plt.scatter(dendro_metrics[:,0].astype(float), dendro_metrics[:,1].astype(float))
    plt.xlabel('Automatic (in meters)')
    plt.ylabel('Manual (in meters)')
#    # Pearson
    #plt.annotate('{0}'.format('Pearson:'),(15,25),size=10)
    #plt.annotate('{:.4f}'.format(pearson_corr[0]),(22,25),size=10)
#    
#    # Spearman
    #plt.annotate('{0}'.format('Spearman: '),(15,23),size=10)
    #plt.annotate('{:.4f}'.format(spearman_corr[0]),(22,23),size=10)
    plt.title(plot_title)
    
    if (not out_file is None):
        plt.savefig(out_file,dpi=300,bbox_inches='tight')
    
    return pearson_corr, spearman_corr, mean_abs_error

input_folder = 'example/validation/with constraints'
dendro_file = os.path.join(os.path.dirname(os.getcwd()), input_folder + '/dendrometric_features_new.txt')

dendro_metrics = np.loadtxt(dendro_file,
                            delimiter=',', dtype=object)

#####################################################################################################################
# Tree height
tree_height = dendro_metrics[:, [1,3]].astype(float)

idx_nan, _ = np.where(np.isnan(tree_height[:,0:]))
tree_height = np.delete(tree_height, idx_nan, axis=0)

p, s, mae = get_correlation(tree_height, plot_title='Height of the tree',
                            out_file=os.path.join(os.path.dirname(os.getcwd()), input_folder + '/height_correlation_all.png'))

idx_non_zero, = np.where(tree_height[:,0] == 0)
tree_height = np.delete(tree_height, idx_non_zero, axis=0)

p1, s1, mae1 = get_correlation(tree_height, plot_title='Height of the tree',
                               out_file=os.path.join(os.path.dirname(os.getcwd()), input_folder + '/height_correlation_non_zero.png'))
#####################################################################################################################


#####################################################################################################################
# Diameter of the crown
crow_diameter = dendro_metrics[:, [2,-1]].astype(float)

idx_nan, _ = np.where(np.isnan(crow_diameter[:,0:]))
crow_diameter = np.delete(crow_diameter, idx_nan, axis=0)

p, s, mae = get_correlation(crow_diameter, plot_title='Crown diameter',
                            out_file=os.path.join(os.path.dirname(os.getcwd()), input_folder + '/crown_diam_correlation_all.png'))

idx_non_zero, = np.where(crow_diameter[:,0] == 0)
crow_diameter = np.delete(crow_diameter, idx_non_zero, axis=0)

p1, s1, mae1 = get_correlation(crow_diameter, plot_title='Crown diameter',
                               out_file=os.path.join(os.path.dirname(os.getcwd()), input_folder + '/crown_diam_correlation_non_zero.png'))
#####################################################################################################################