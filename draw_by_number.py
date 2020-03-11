# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:30:01 2019

@author: Li Xiang
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint

fp='./documents/'
sns.set(font_scale=1.2)
datasets=['laptop_cpu','laptop_hd','laptop_gpu','laptop_screen','laptop_ram']
filetype=['_eval_by_label.txt','_test_labels_count.txt','_train_labels_count.txt']

r1_dataframe_lst=[]
r5_dataframe_lst=[]
f1_1_dataframe_lst=[]
f1_5_dataframe_lst=[]
for datas in datasets:
    
    with open(fp+datas+filetype[0]) as f1:
        evaluation=f1.read()
    with open(fp+datas+filetype[2]) as f2:
        labels_map_count=eval(f2.read())
#    train_count={k: v for k, v in sorted(train_num.items(), key=lambda item: item[1])}
    with open(fp+datas+filetype[1]) as f3:
        test_labels=eval(f3.read())
        
    aa=evaluation.split('\n')
#    labels_map_count=[l.split(':')[1] for l in aa if(l.startswith('label'))]
    f1s_top1=[eval(l.split(':')[1])[0] for l in aa if(l.startswith('f1'))]
    f1s_top5=[eval(l.split(':')[1])[4] for l in aa if(l.startswith('f1'))]
    recalls_top1=[eval(l.split(':')[1])[0] for l in aa if(l.startswith('recall'))]
    recalls_top5=[eval(l.split(':')[1])[4] for l in aa if(l.startswith('recall'))]
    
    
    for i,label in enumerate(test_labels):
        r1_dataframe_lst.append([labels_map_count[label],recalls_top1[i],datas.split('_')[1]])
    for i,label in enumerate(test_labels):
        r5_dataframe_lst.append([labels_map_count[label],recalls_top5[i],datas.split('_')[1]])
    for i,label in enumerate(test_labels):
        f1_1_dataframe_lst.append([labels_map_count[label],f1s_top1[i],datas.split('_')[1]])
    for i,label in enumerate(test_labels):
        f1_5_dataframe_lst.append([labels_map_count[label],f1s_top5[i],datas.split('_')[1]])



#    dataframe_lst.append[]
#print(r1_dataframe_lst)
pprint(r5_dataframe_lst)
#print(f1_1_dataframe_lst)
#print(f1_5_dataframe_lst)
    
r1_df=pd.DataFrame(r1_dataframe_lst,columns =['number of samples in trainset','recall@1','class'])
r5_df=pd.DataFrame(r5_dataframe_lst,columns =['number of samples in trainset','recall@5','class'])
f1_1_df=pd.DataFrame(f1_1_dataframe_lst,columns =['number of samples in trainset','f1@1','class'])
f1_5_df=pd.DataFrame(f1_5_dataframe_lst,columns =['number of samples in trainset','f1@5','class'])

plt.figure(figsize=(10, 6))

sns.scatterplot(x="number of samples in trainset", y="recall@1",\
                hue="class", data=r1_df)#.set_title('Recall@1')

plt.figure(figsize=(10, 6))

sns.scatterplot(x="number of samples in trainset", y="recall@5",\
                hue="class", data=r5_df)#.set_title('Recall@1')
#    break

plt.figure(figsize=(10, 6))

sns.scatterplot(x="number of samples in trainset", y="f1@1",\
                hue="class", data=f1_1_df)#.set_title('Recall@1')
#    break

plt.figure(figsize=(10, 6))

sns.scatterplot(x="number of samples in trainset", y="f1@5",\
                hue="class", data=f1_5_df)#.set_title('Recall@1')
#    break
    