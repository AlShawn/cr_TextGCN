# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:08:24 2020

@author: Li Xiang

20% long and 80% short evaluation

#----run train.py first to get test_labels,test_pred-------------

# find text length and index in file data/corpus/laptop_xx_shuffle.txt 
# each laptop_xx_shuffle.txt  have different index!
# in the test sentences, which indexes are long / short
# then evaluate accordingly
# find out which index are used for testing in train.py

"""


import os
import numpy as np

import sys
import func_eval

#datasets=['laptop_cpu','laptop_hd','laptop_gpu','laptop_screen','laptop_ram']
#dataset = datasets[0]


# get text_lst and text_length_lst
fpath='./data/corpus/'
file_text=dataset+'_shuffle.txt'
with open(fpath+file_text,'r') as f:
    text_lst=f.read().splitlines()

text_length_lst=[]
for each in text_lst:
    text_length_lst.append(len(each.split()))
    

# get index_label_lst
flabelpath='./data/'
file_label_ind=dataset+'_shuffle.txt'
with open(flabelpath+file_text,'r') as f:
    b=f.read().splitlines()
index_label_lst=[]
for i,each in enumerate(b):
    index_label_lst.append(each.split())
    
# get indexes of long/short
test_inds=[]
for i,each in enumerate(index_label_lst):
    if(each[1]=='test'):
        test_inds.append(int(each[0]))
        
test_length_lst=[]
for ind in test_inds:
    test_length_lst.append(text_length_lst[ind])
        
l=len(test_length_lst)
long_short_split_length=sorted(test_length_lst,reverse=True)[int(0.2*l)]    
    
#test_long_short_lst: mark the test sample is long or short 
test_long_short_lst=[]
for l in test_length_lst:
    if(l>=long_short_split_length):
        test_long_short_lst.append('long')
    else:
        test_long_short_lst.append('short')
        
    
#---------get long/short evaluation---------------------------------------
#test_labels,test_pred
long_test_labels=[]
long_test_pred=[]
short_test_labels=[]
short_test_pred=[]


for ind,each in enumerate(test_long_short_lst):
    if(each=='long'):
        long_test_labels.append(test_labels[ind])
        long_test_pred.append(test_pred[ind])
    else:
        short_test_labels.append(test_labels[ind])
        short_test_pred.append(test_pred[ind])
    
long_test_pred=np.array(long_test_pred)  
short_test_pred=np.array(short_test_pred)    
    

#---------------evaluate long and short------------------------------
print('-------------------------------------------------------------')
print('---exper_param:',dataset)
print('--long--')

y_real=[]
for i,each in enumerate(long_test_labels):
    y_real.append(np.array([each]))
y_real=np.array(y_real)    


print("ncdg:")
i=0
ndcgs=[]
top_k=5
while i < top_k:
    
    y_pred = long_test_pred[:, 0:i+1]
    i = i+1

    ndcg_i = func_eval._NDCG_score(y_pred,y_real)
    ndcgs.append(ndcg_i)

    print(ndcg_i)

#sys.exit(0)  

print("precision:")
i = 0
precisions = []

while i < top_k:
    
    y_pred = long_test_pred[:, 0:i+1]
    i = i+1

    precision = func_eval._precision_score(y_pred,y_real)
    precisions.append(precision)

    print(precision)

print("recall:")
i = 0
recalls = []
while i < top_k:
    
    y_pred = long_test_pred[:,  0:i+1]

    i = i+1   
    recall = func_eval.new_recall(y_pred, y_real)
    recalls.append(recall)

    print(recall)



print('\n--short--')

y_real=[]
for i,each in enumerate(short_test_labels):
    y_real.append(np.array([each]))
y_real=np.array(y_real)    


print("ncdg:")
i=0
ndcgs=[]
top_k=5
while i < top_k:
    
    y_pred = short_test_pred[:, 0:i+1]
    i = i+1

    ndcg_i = func_eval._NDCG_score(y_pred,y_real)
    ndcgs.append(ndcg_i)

    print(ndcg_i)

#sys.exit(0)  

print("precision:")
i = 0
precisions = []

while i < top_k:
    
    y_pred = short_test_pred[:, 0:i+1]
    i = i+1

    precision = func_eval._precision_score(y_pred,y_real)
    precisions.append(precision)

    print(precision)

print("recall:")
i = 0
recalls = []
while i < top_k:
    
    y_pred = short_test_pred[:,  0:i+1]

    i = i+1   
    recall = func_eval.new_recall(y_pred, y_real)
    recalls.append(recall)

    print(recall)    
    
    
    
    