# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:43:05 2020

@author: DANILO
"""

import os
import argparse
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def get_main_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_file', type=str, help='Name of the file with the dendrometric measurements')
    
    parser.add_argument('-output_folder', type=str, default=None, help='Folder where the output files should be saved. Default is None')
    parser.add_argument('-save_plots', action='store_true', default=False, help='Indicates whether the plots should be saved. Default is False')
    
    return parser.parse_args()

def print_correlation(p,s,alpha):
    print('Pearson correlation: ', p[0], '. p-value: ',p[1])
    print('Spearman correlation: ', s[0], '. p-value: ',s[1])
    
    if (p[1] < alpha and s[1] < alpha):
        print('Reject null hypothesis. The correlation is statiscally significant at 5%')
    else:
        print('Accept null hypothesis. The correlation is not statiscally significant at 5%')        

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

def main(args):
    dendro_file = args.input_file
    output_folder = args.output_folder
    save_plots = args.save_plots    
    
    dendro_metrics = None
    
    alpha = 0.05
    
    try:
        dendro_metrics = np.loadtxt(dendro_file, delimiter=',', dtype=object)
    except:
        raise Exception('An error occurred during the opening of the input file')

    #####################################################################################################################
    # Tree height
    tree_height = dendro_metrics[:, [1,3]].astype(float)
    
    idx_nan, _ = np.where(np.isnan(tree_height[:,0:]))
    tree_height = np.delete(tree_height, idx_nan, axis=0)
    
    if (save_plots):
        plot_name = os.path.join(output_folder,'height_correlation_all.png')
    else:
        plot_name = None
    
    p, s, mae = get_correlation(tree_height, plot_title='Height of the tree',
                                out_file=plot_name)
    
    print('Tree height - all values')
    print_correlation(p,s,alpha)
    
    idx_non_zero, = np.where(tree_height[:,0] == 0)
    tree_height = np.delete(tree_height, idx_non_zero, axis=0)

    if (save_plots):
        plot_name = os.path.join(output_folder,'height_correlation_non_zero.png')
    else:
        plot_name = None    
    
    p1, s1, mae1 = get_correlation(tree_height, plot_title='Height of the tree',
                                   out_file=plot_name)
    
    print('\n\n')
    
    print('Tree height - non-zero values')
    print_correlation(p1,s1,alpha)    
    #####################################################################################################################
    
    
    #####################################################################################################################
    # Diameter of the crown
    crow_diameter = dendro_metrics[:, [2,-1]].astype(float)
    
    idx_nan, _ = np.where(np.isnan(crow_diameter[:,0:]))
    crow_diameter = np.delete(crow_diameter, idx_nan, axis=0)
    
    if (save_plots):
        plot_name = os.path.join(output_folder,'crown_diam_correlation_all.png')
    else:
        plot_name = None    
    
    p, s, mae = get_correlation(crow_diameter, plot_title='Crown diameter',
                                out_file=plot_name)
    
    print('\n\n')
    
    print('Crown diameter - all values')
    print_correlation(p,s,alpha)    
    
    idx_non_zero, = np.where(crow_diameter[:,0] == 0)
    crow_diameter = np.delete(crow_diameter, idx_non_zero, axis=0)
    
    if (save_plots):
        plot_name = os.path.join(output_folder,'crown_diam_correlation_non_zero.png')
    else:
        plot_name = None        
    
    p1, s1, mae1 = get_correlation(crow_diameter, plot_title='Crown diameter',
                                   out_file=plot_name)
    
    print('\n\n')
    
    print('Crown diameter - non-zero values')
    print_correlation(p1,s1,alpha)
    #####################################################################################################################

if __name__ == '__main__':
    args = get_main_args()
    main(args)