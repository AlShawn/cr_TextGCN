# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:47:25 2019

@author: Li Xiang

run train.py first

output file stored in folder ./document/
"""
import sys
import numpy as np
from pprint import pprint
import func_eval
#---------------variables to be used in label frequency experiment------------
#----train_mask,y_train----
#----test_labels,test_pred,test_mask,-------------

train_labels=[y_train[i] for i, x in enumerate(train_mask) if x]
train_labels=list(np.argmax(train_labels,axis=1))

train_labels_unique=set(train_labels)

num_labels_train_dic={}
for l in train_labels_unique:
    num_labels_train_dic[l]=train_labels.count(l)
    
pprint(num_labels_train_dic)


num_labels_test_dic={}
test_labels_unique=set(test_labels)
for l in test_labels_unique:
    num_labels_test_dic[l]=test_labels.count(l)
    
pprint(num_labels_test_dic)

#------real label: test_labels--------------------
#------predicted labels: test_pred-----------------

test_label_map_ind={}
for l in num_labels_test_dic:
    nums=[ind for ind,label in list(enumerate(test_labels)) if label==l]
    test_label_map_ind[l]=nums
top_k=5

print('class:',dataset)
for l in test_label_map_ind:
#    test
    real_labels=np.array(test_labels)[test_label_map_ind[l]]
    real_labels=np.array([np.array([l]) for l in real_labels])
    pred_labels=test_pred[test_label_map_ind[l]]
    print('\nlabel:',l)
    
    

    print("precision:")
    i = 0
    precisions = []
    
    while i < top_k:
        
        y_pred = pred_labels[:, 0:i+1]
        i = i+1
    
        precision = func_eval._precision_score(y_pred,real_labels)
        precisions.append(precision)
    
        print(precision)
    
    print("recall:")
    i = 0
    recalls = []
    while i < top_k:
        
        y_pred = pred_labels[:,  0:i+1]
    
        i = i+1   
        recall = func_eval.new_recall(y_pred, real_labels)
        recalls.append(recall)
    
        print(recall)
        
    print("f1:")
    f1s=[]
    for i in range(len(precisions)):
        if(precisions[i]+recalls[i]!=0):
            f1=((precisions[i]*recalls[i])/(precisions[i]+recalls[i]))
            print(f1)
        else:
            f1=0.0
            print(0.0)
        f1s.append(f1)
    
    with open('./documents/'+dataset+'_eval_by_label.txt','a') as f:
        f.write('\n\nlabel:'+str(l))
        f.write('\nprecision:')
        f.write(str(precisions))
        f.write('\nrecall:')
        f.write(str(recalls))
        f.write('\nf1:')
        f.write(str(f1s))
        
    
with open('./documents/'+dataset+'_train_labels_count.txt','w') as f:
    f.write(str(num_labels_train_dic))


with open('./documents/'+dataset+'_test_labels_count.txt','w') as f:
    f.write(str(num_labels_test_dic))







