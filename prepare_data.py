# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:30:53 2019

@author: Li Xiang

change the dataset's format to adapt to the textGCN model

create 5 files: 'laptop_cpu.txt','laptop_hd.txt','laptop_gpu.txt',
                'laptop_screen.txt','laptop_ram.txt'

* use relabeled cpu version

"""

import pandas as pd
import numpy as np
from func import get_new_cpu_label
from func import get_labels
import sys


review_ds=pd.read_csv('../data/original_dataset.csv',encoding="ISO-8859-1")
needs_ds=pd.read_csv('../data/generated_reviews.csv',encoding='utf-8')

#----------------- only extract reviews with valid asins -----------------
#--------------------append all text to review_lst------------------------
asin_lst=[]
fpath='..//data//'+'asin_map_'+'ram'+'_rank.npy'    
asin_map_labels=np.load(fpath).item()
for asin in list(asin_map_labels.keys()):
    asin_lst.append(asin)

review_lst=review_ds.loc[review_ds['asin'].isin(asin_lst), 'reviews'].tolist()
needs_lst=needs_ds.loc[needs_ds['asin'].isin(asin_lst), 'reviews'].tolist()

trainset_len=len(review_lst)
testset_len=len(needs_lst)
print("length of reviews:", len(review_lst))
print("length of needs:", len(needs_lst))

review_lst.extend(needs_lst)

#------------- write useful reviews and needs to dataset file -----------------
#------------- create 5 files because they share the same text--------------
corpus_str='\n'.join(review_lst)

#dataset_name='laptop'
dataset_name_lst=['laptop_cpu','laptop_hd','laptop_gpu','laptop_screen','laptop_ram']
for dataset_name in dataset_name_lst:
    fpath='./data/corpus/' + dataset_name + '.txt'
    with open(fpath,'w',encoding="utf-8") as f:
        f.write(corpus_str)

#sys.exit(0)

#-----------for each class, generate different label set------------------------

cpu_labels=[]
hd_labels=[]
screen_labels=[]
gpu_labels=[]
ram_labels=[]

cpu_asin_map_labels=get_new_cpu_label()
hd_asin_map_labels=get_labels('hd')
screen_asin_map_labels=get_labels('screen')
gpu_asin_map_labels=get_labels('gpu')
ram_asin_map_labels=get_labels('ram')

for each_asin in review_ds.loc[review_ds['asin'].isin(asin_lst), 'asin']:
    if(each_asin in asin_lst):
        cpu_labels.append(cpu_asin_map_labels[each_asin][0])
        hd_labels.append(hd_asin_map_labels[each_asin][0])
        screen_labels.append(screen_asin_map_labels[each_asin][0])
        gpu_labels.append(gpu_asin_map_labels[each_asin][0])
        ram_labels.append(ram_asin_map_labels[each_asin][0])

for each_asin in needs_ds.loc[needs_ds['asin'].isin(asin_lst), 'asin']:
    if(each_asin in asin_lst):
        cpu_labels.append(cpu_asin_map_labels[each_asin][0])
        hd_labels.append(hd_asin_map_labels[each_asin][0])
        screen_labels.append(screen_asin_map_labels[each_asin][0])
        gpu_labels.append(gpu_asin_map_labels[each_asin][0])
        ram_labels.append(ram_asin_map_labels[each_asin][0])
    
    
train_or_test_list=['train' for i in range(trainset_len)]   
train_or_test_list.extend(['test' for i in range(testset_len)])


#------------- write useful reviews and needs to dataset file -----------------

#labels=[cpu_labels,hd_labels,screen_labels,gpu_labels,ram_labels]
name_map_labels={}
name_map_labels['cpu']=cpu_labels
name_map_labels['hd']=hd_labels
name_map_labels['screen']=screen_labels
name_map_labels['gpu']=gpu_labels
name_map_labels['ram']=ram_labels

for name in name_map_labels:
    meta_data_list = []
    
    for i in range(len(review_lst)):
        meta = str(i) + '\t' + train_or_test_list[i] + '\t' + str(name_map_labels[name][i])
        meta_data_list.append(meta)
    
    meta_data_str = '\n'.join(meta_data_list)
    
    fpath='./data/laptop_'+name+ '.txt'
    with open(fpath,'w',encoding="utf-8") as f:
        f.write(meta_data_str)



#-------------------------------original-------------------------------------

#dataset_name = 'own'
#sentences = ['Would you like a plain sweater or something else?​', 'Great. We have some very nice wool slacks over here. Would you like to take a look?']
#labels = ['Yes' , 'No' ]
#train_or_test_list = ['train', 'test']
#
#
#meta_data_list = []
#
#for i in range(len(sentences)):
#    meta = str(i) + '\t' + train_or_test_list[i] + '\t' + labels[i]
#    meta_data_list.append(meta)
#
#meta_data_str = '\n'.join(meta_data_list)
#
#f = open('./data/' + dataset_name + '.txt', 'w')
#f.write(meta_data_str)
#f.close()
#
#corpus_str = '\n'.join(sentences)
#
#f = open('./data/corpus/' + dataset_name + '.txt', 'w',encoding="utf-8")
#f.write(corpus_str)
#f.close()

