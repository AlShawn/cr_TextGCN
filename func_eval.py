# -*- coding: utf-8 -*-
"""
elmo

Created on Tue Feb 19 12:31:27 2019

@author: Li Xiang
"""

import numpy as np
from numpy import log2

def new_recall(y_pred,y_real):

    recall = 0
    total_recall=0
    num = y_real.shape[0]*y_real.shape[1]
    
#    for each row, if there is matched class, then it's a count
#    simply check each row of y_pred and y_real if their intersection is true
#   at each top-k, the number of relevant has 
    for i in range(y_pred.shape[0]):
        overlap_len=len(set(y_pred[i,:]).intersection(y_real[i,:]))
        if(overlap_len!=0):
            total_recall=total_recall+overlap_len
            
    recall=total_recall/num
    return recall

def _DCG_cal(each_row):
    dcg_cur=each_row[0]
    r=2
    while r<=len(each_row):
        dcg_cur+=each_row[r-1]/log2(r)
        r+=1
    return dcg_cur
        
#y_top_K,labels_to_eval
def _NDCG_score(y_pred,y_real):
    ndcg=0
    rows,cols=y_pred.shape
    score_matrix=np.zeros(y_pred.shape)
    for i in range(rows):
        for j in range(cols):
            if(y_pred[i][j] in y_real[i]):
                score_matrix[i][j]=1
            else:
                score_matrix[i][j]=0
                
    ndcg=[]
    for each_row in score_matrix:
#        ndcg = rel_0+sum(rel_i/log2(i))
        dcg_cur=_DCG_cal(each_row)
        idcg_cur=_DCG_cal(sorted(each_row,reverse=True))
        if(idcg_cur!=0):
            ndcg.append(dcg_cur/idcg_cur)
        else:
            ndcg.append(0)
    ave_ndcg=sum(ndcg)/len(ndcg)
    
    return ave_ndcg

def _precision_score(y_pred, y_real):
    precision = 0

    num = 0
    i = 0
#for precision computation
#simply check if each top-k results of sample match 
#if match, we plus one to num
#the final result is num/total# of predicted results
    while i < y_pred.shape[1]:
#        lst is each col in y_test
        lst = y_pred[:, i].tolist()
        j = 0
        while  j < len(lst):
            if lst[j] in y_real[j].tolist():
                num = num + 1
            j = j + 1
            
        i = i + 1
        
    precision = num / (len(y_real) * y_pred.shape[1])#total_row*col count

    return precision

def new_recall_score(y_pred,y_real):
    
    recall = 0
    total_recall=0
    num = y_real.shape[0]
    
#    for each row, if there is matched class, then it's a count
#    simply check each row of y_pred and y_real if their intersection is true
#   at each top-k, the number of relevant has 
    for i in range(y_pred.shape[0]):
        if(set(y_pred[i,:]).intersection(y_real[i,:])):
            total_recall=total_recall+1
            
    recall=total_recall/num
    return recall