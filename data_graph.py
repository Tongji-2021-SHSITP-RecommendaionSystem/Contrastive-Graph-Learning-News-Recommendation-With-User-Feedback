# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 14:22:43 2021

@author: 19526
"""
import scipy.sparse as sp
import numpy as np
# import tensorflow.keras as kr
import pandas as pd
import pickle
def news_process(news):
    content=[]
    category=[]
    # entity=[]
    for i in range(len(news)):
        content_one=news.title[i].strip('[').strip(']').split(',')
        content_one.extend(news.abstract[i].strip('[').strip(']').split(','))
        category_one=news.category[i]
        content.append(list(map(int,content_one)))
        category.append(category_one)
    news_content=dict(zip(list(news.id),content))
    news_category=dict(zip(list(news.id),category))
    return news_content,content,news_category,category

def batch_iter(users,news_id,max_length=100,click_size=30,candidate_size=10,real_num=3):
    news_dict=dict() # 用新闻id得到在list的第几个
    news_list=[] # 存新闻内
    click_batch=[]
    for i in range(len(users)):
        click_id=str(users.history.iloc[i]).split(" ")
        click=[]
        len_click_id=len(click_id)# if len(click_id)<=click_size+real_num else click_size+real_num
        if(len_click_id<candidate_size+real_num):
            continue
        # print("len_click_id",len_click_id)
        for j in range(len_click_id):
            if(click_id[j] in news_id):
                if (click_id[j] not in news_dict.keys()):
                    if (len(news_id[click_id[j]])<max_length):
                        news_id[click_id[j]].extend([0]*(max_length-len(news_id[click_id[j]])))
                    else:
                        news_id[click_id[j]]=news_id[click_id[j]][0:max_length]
                    news_list.append(news_id[click_id[j]])
                    news_dict[click_id[j]]=len(news_list)-1
                    click.append(len(news_list)-1)
                else:
                    click.append(news_dict[click_id[j]])
        click_batch.append(click)
    R = sp.dok_matrix((len(click_batch), len(news_list)), dtype=np.float32)
    print("user:",len(click_batch), "news:",len(news_list))
    for i in range(len(click_batch)):
        for j in range(len(click_batch[i])):
            R[i,click_batch[i][j]]=1.0
    return R

users=pd.read_csv("E:/2019sitp/data/endata/train_plus.tsv",sep='\t',index_col=False)
news=pd.read_csv("E:/2019sitp/data/endata/news_parsed.tsv",sep='\t',index_col=False)
news_content,content,news_entity,entity=news_process(news)
R = batch_iter(users,news_content)
f=open('./seq_mat1','wb')
pickle.dump(R.tocsr(),f)
print("save")