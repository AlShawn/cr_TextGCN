# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:05:52 2019

@author: Li Xiang
"""

import numpy as np

def get_labels(exp_param):
    if(exp_param in ['ram','gpu','hd','screen']):
        fpath='..//data//'+'asin_map_'+exp_param+'_rank.npy'    
        asin_map_labels=np.load(fpath).item()
    return asin_map_labels
    

def get_new_cpu_label():
    """
    give cpu new label 0-9, in order to solve the problem of missing values
    
    cpu label mapping table:
        Description	                Label
        Intel Celeron/ADM A (0, 2GHz)	0
        Intel Celeron/ADM A ([2, 3)GHz	1
        Intel Celeron/ADM A ([3, )GHz	2
        Intel i3 (0, 2.4) GHz	         3
        Intel i3 [2.4, ) GHz	         4
        Intel i5 (0, 2] GHz	         5
        Intel i5 (2, 3) GHz	         6
        Intel i5 [3, ) GHz	            7
        Intel i7 (0, 2] GHz	         6
        Intel i7 (2, 3] GHz	         7
        Intel i7 [3, ) GHz	            8
        Others	                        9    
    """
    
    cpu_tech_file='..//data//amazon_tech_cpus_1207.json'
    asin_map_labels = {}
    with open(cpu_tech_file, 'r') as f1:
        for line in f1:
#            if '+' in line:
                #ind += 1
            asin = line.split(':')[0].strip()
            cpu_model = line.split(':')[1].strip()
            md_lst=cpu_model.split()
            Ghz=0
            #get the GHz number
            if(len(md_lst)>1 and md_lst[1]=='GHz'):
                Ghz=float(md_lst[0])
            if(('Celeron' in cpu_model)or('AMD' in cpu_model)):
                if(Ghz<2):
                    label=[0]
                elif(Ghz<3):
                    label=[1]
                else:
                    label=[2]
            elif('i3' in cpu_model):
                if(Ghz<2.4):
                    label=[3]
                else:
                    label=[4]
            elif('i5' in cpu_model):
                if(Ghz<2):
                    label=[5]
                elif(Ghz<3):
                    label=[6]
                else:
                    label=[7]
            elif('i7' in cpu_model):
                if(Ghz<2):
                    label=[6]
                elif(Ghz<3):
                    label=[7]
                else:
                    label=[8]
            else:
                label=[9]
                
            asin_map_labels[asin]=label
                        
    return asin_map_labels          
    
                    
                