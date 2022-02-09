# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 14:22:43 2021

@author: 19526
"""
import scipy.sparse as sp
import numpy as np
# import torch.nn as nn
# import tensorflow.keras as kr
import pandas as pd
import random

def news_process(news):
    content=[]
    category=[]
    # entity=[]
    for i in range(len(news)):
        content_one=news.title[i].strip('[').strip(']').split(',')
        content_one.extend(news.abstract[i].strip('[').strip(']').split(','))
        category_one=news.category[i]
        # entity_one=news.title_entities[i].strip('[').strip(']').split(',')
        # entity_one.extend(news.abstract_entities[i].strip('[').strip(']').split(','))
        # print(list(map(int,content_one)))
        content.append(list(map(int,content_one)))
        category.append(category_one)
        # entity.append(list(map(int,entity_one)))
    news_content=dict(zip(list(news.id),content))
    news_category=dict(zip(list(news.id),category))
    # news_entity=dict(zip(list(news.id),entity))
    return news_content,content,news_category,category# news_entity,entity

def batch_iter(users,news_id,news,news_category,category,category_num=50,batch_size=64,max_length=100,click_size=30,candidate_size=10,real_num=3,refuse_num=5):
    batch_count=0
    click_batch=[]
    clickrefuse_batch=[]
    candidate_batch=[]
    # click_entity_batch=[]
    # candidate_entity_batch=[]
    news_dict=dict() # 用新闻id得到在list的第几个
    news_list=[] # 存新闻内
    news_list_length=(click_size+real_num+refuse_num)*batch_size+1
    # 标记新闻属于哪个用户，使得候选新闻不出现历史新闻
    news_list_used=[[] for ii in range(news_list_length)]
    # print("length",news_list_length)
    # print("candidate",candidate_size)
    # print("real_num",real_num)
    # print(news_category)
    news_adj=[[0 for ii in range(news_list_length)]for jj in range(news_list_length)]# [[0]*news_list_length]*news_list_length
    news_c_adj=[]# [[0 for ii in range(news_list_length)]for jj in range(news_list_length)]
    category_adj=[[0 for ii in range(category_num)]for jj in range(category_num)]
    news_name_list=list(news_id)
    for i in range(len(users)):
        if(batch_count==batch_size):
            batch_count=0
            click_batch=[]
            clickrefuse_batch=[]
            candidate_batch=[]
            # click_entity_batch=[]
            # candidate_entity_batch=[]
            news_list=[]
            news_dict=dict()
            news_list_used=[[] for ii in range(news_list_length)]
            news_adj=[[0 for ii in range(news_list_length)]for jj in range(news_list_length)]
            news_c_adj=[]# [[0 for ii in range(news_list_length)]for jj in range(news_list_length)]
            category_adj=[[0 for ii in range(category_num)]for jj in range(category_num)]
            
        click_id=str(users.history.iloc[i]).split(" ")
        refuse_id=str(users.click.iloc[i]).split(" ")
        click=[]
        clickrefuse=[]
        # click_entity=[]
        preno=-1
        preid=""
        len_click_id=len(click_id) if len(click_id)<=click_size+real_num else click_size+real_num
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
                # click.append(news_id[click_id[j]])
                else:
                    click.append(news_dict[click_id[j]])
                # print(news_dict[click_id[j]])
                # print(news_list_used)
                news_list_used[news_dict[click_id[j]]].extend([batch_count%batch_size])
                # click_entity.append(news_entity[click_id[j]])
                # print(preno,len(category))
                if(preno!=-1 and len_click_id-j>3):
                    news_adj[preno][news_dict[click_id[j]]]=1
                    news_adj[news_dict[click_id[j]]][preno]=1
                    # print(news_category[click_id[j]],news_category[preid])
                    # print(len(category_adj))
                    category_adj[news_category[click_id[j]]][news_category[preid]]=1
                    category_adj[news_category[preid]][news_category[click_id[j]]]=1
                news_c_adj.append((news_category[click_id[j]],news_dict[click_id[j]]))
                # news_c_adj[news_dict[click_id[j]]][news_category[click_id[j]]]=1
                    # print(news_adj)
                preno=news_dict[click_id[j]]
                preid=click_id[j]
            # else:
                # j-=1
        preno=-1
        preid=""
        for j in range(len(refuse_id)):
            # print(refuse_id[j])
            if(refuse_id[j][-1]=='0' and refuse_id[j][:-2] in news_id):
                # print(refuse_id[j][:-2])
                if (refuse_id[j][:-2] not in news_dict.keys()):
                    if (len(news_id[refuse_id[j][:-2]])<max_length):
                        news_id[refuse_id[j][:-2]].extend([0]*(max_length-len(news_id[refuse_id[j][:-2]])))
                    else:
                        news_id[refuse_id[j][:-2]]=news_id[refuse_id[j][:-2]][0:max_length]
                    news_list.append(news_id[refuse_id[j][:-2]])
                    news_dict[refuse_id[j][:-2]]=len(news_list)-1
                    clickrefuse.append(len(news_list)-1)
                # click.append(news_id[click_id[j]])
                else:
                    clickrefuse.append(news_dict[refuse_id[j][:-2]])
                # print(news_dict[click_id[j]])
                # print(news_list_used)
                news_list_used[news_dict[refuse_id[j][:-2]]].extend([batch_count%batch_size])
                # click_entity.append(news_entity[refuse_id[j][:-2]])
                if(preno!=-1 and len_click_id-j>3):
                    news_adj[preno][news_dict[refuse_id[j][:-2]]]=1
                    news_adj[news_dict[refuse_id[j][:-2]]][preno]=1    
                    category_adj[news_category[refuse_id[j][:-2]]][news_category[preid]]=1
                    category_adj[news_category[preid]][news_category[refuse_id[j][:-2]]]=1
                news_c_adj.append((news_category[refuse_id[j][:-2]],news_dict[refuse_id[j][:-2]]))
                preno=news_dict[refuse_id[j][:-2]]
                preid=refuse_id[j][:-2]
                if(len(clickrefuse)>=refuse_num):
                    break
        
                
        batch_count+=1
        candidate=click[-real_num:]
        # candidate_entity=click_entity[-real_num:]
        # print(candidate)
        '''
        for j in range(candidate_size-real_num):
            randno=random.randint(0,120958)
            candidate.append(news[randno])
            candidate_entity.append(entity[randno])
        '''
        click=click[0:-real_num]
        # click_entity=click_entity[0:-real_num]
        
        # news_list=nn.utils.rnn.pad_sequence(news_list, batch_first=False, padding_value=0)
        # news_list=news_list[:][0:max_length]
        # click= kr.preprocessing.sequence.pad_sequences(click[0:click_size],max_length)
        # click_entity= kr.preprocessing.sequence.pad_sequences(click_entity[0:click_size],max_length)
        if(len(click)<click_size):
            # pad=np.zeros(shape=(click_size-len(click),max_length),dtype=np.int)
            # click=np.concatenate((click,pad))
            # click_entity=np.concatenate((click_entity,pad))
            click.extend([-1]*(click_size-len(click)))
        if(len(clickrefuse)<refuse_num):
            clickrefuse.extend([-1]*(refuse_num-len(clickrefuse)))
        click_batch.append(click)
        clickrefuse_batch.append(clickrefuse)
        # click_entity_batch.append(click_entity)
        candidate_batch.append(candidate)
        # candidate_entity_batch.append(candidate_entity)
        # candidate_batch.append( kr.preprocessing.sequence.pad_sequences(candidate,max_length))
        # candidate_entity_batch.append( kr.preprocessing.sequence.pad_sequences(candidate_entity,max_length))
        if(batch_count==batch_size):
            # 先将新闻列表填满，再随机抽取候选新�?
            upno=len(news_name_list)-1
            while(len(news_list)<news_list_length):
                randno=random.randint(0,upno)
                if(news_name_list[randno] not in news_dict):
                    if (len(news_id[news_name_list[randno]])<=max_length):
                        news_id[news_name_list[randno]].extend([0]*(max_length-len(news_id[news_name_list[randno]])))
                    else:
                        news_id[news_name_list[randno]]=news_id[news_name_list[randno]][0:max_length]
                    news_list.append(news_id[news_name_list[randno]])
                    news_dict[news_name_list[randno]]=len(news_list)-1
            # 填充候选新�?
            for k in range(batch_size):
                while(len(candidate_batch[k])<candidate_size):
                    randno=random.randint(0,news_list_length-1)
                    # print(randno)
                    if(k not in news_list_used[randno]):
                        candidate_batch[k].extend([randno])
            print(news_dict)      
            yield np.array(news_adj),np.array(news_list),np.array(click_batch),np.array(clickrefuse_batch),np.array(candidate_batch),news_c_adj,np.array(category_adj)
            #click_entity_batch,candidate_batch,candidate_entity_batch,candidate_batch[0]
    return

def news_process_zh(news):
    content=[]
    category=[]
    # entity=[]
    for i in range(len(news)):
        
        content_one=news.content[i].split(' ')
        category_one=news.category[i].strip(' ').strip('"').split(',')
        content.append(list(map(int,content_one)))
        category.append(category_one[0])
    news_content=dict(zip(list(news.newsid),content))
    news_category=dict(zip(list(news.newsid),category))
    return news_content,content,news_category,category

def batch_iter_zh(users,news_id,news,news_category,category,category_num=800,batch_size=64,max_length=100,click_size=30,candidate_size=10,real_num=3,refuse_num=5):
    batch_count=0
    click_batch=[]
    clickrefuse_batch=[]
    candidate_batch=[]
    news_dict=dict() # 用新闻id得到在list的第几个
    cat_dict=dict()
    news_list=[] # 存新闻内�?
    news_list_length=(click_size+real_num+refuse_num)*batch_size+1
    # 标记新闻属于哪个用户，使得候选新闻不出现历史新闻
    news_list_used=[[] for ii in range(news_list_length)]
    news_adj=[[0 for ii in range(news_list_length)]for jj in range(news_list_length)]# [[0]*news_list_length]*news_list_length
    news_c_adj=[]
    category_adj=[[0 for ii in range(category_num)]for jj in range(category_num)]
    news_name_list=list(news_id)
    for i in range(len(users)):
        if(batch_count==batch_size):
            batch_count=0
            click_batch=[]
            clickrefuse_batch=[]
            candidate_batch=[]
            # click_entity_batch=[]
            # candidate_entity_batch=[]
            news_list=[]
            news_dict=dict()
            cat_dict=dict()
            news_list_used=[[] for ii in range(news_list_length)]
            news_adj=[[0 for ii in range(news_list_length)]for jj in range(news_list_length)]
            news_c_adj=[]# [[0 for ii in range(news_list_length)]for jj in range(news_list_length)]
            category_adj=[[0 for ii in range(category_num)]for jj in range(category_num)]
            
        click_list=str(users.news.iloc[i]).split(",")
        click_list_div=[click_list[j].split("|") for j in range(len(click_list))]
        click=[]
        clickrefuse=[]
        # click_entity=[]
        preno=-1
        preid=""
        len_click_id=0
        for j in range(len(click_list)):
            if(click_list_div[j][2]!='0'):
                len_click_id+=1
        if(len_click_id<candidate_size+real_num):
            continue
        # print("len_click_id",len_click_id)
        for j in range(len(click_list)):
            if(click_list_div[j][2]!='0' and click_list_div[j][0] in news_id):
                if (news_category[click_list_div[j][0]] not in cat_dict.keys()):
                    cat_dict[news_category[click_list_div[j][0]]]=len(cat_dict)
                # print(cat_dict)
                if (click_list_div[j][0] not in news_dict.keys()):
                    if (len(news_id[click_list_div[j][0]])<max_length):
                        news_id[click_list_div[j][0]].extend([0]*(max_length-len(news_id[click_list_div[j][0]])))
                    else:
                        news_id[click_list_div[j][0]]=news_id[click_list_div[j][0]][0:max_length]
                    news_list.append(news_id[click_list_div[j][0]])
                    news_dict[click_list_div[j][0]]=len(news_list)-1
                    click.append(len(news_list)-1)
                else:
                    click.append(news_dict[click_list_div[j][0]])
                news_list_used[news_dict[click_list_div[j][0]]].extend([batch_count%batch_size])
                if(preno!=-1 and len_click_id-j>3):
                    news_adj[preno][news_dict[click_list_div[j][0]]]=1
                    news_adj[news_dict[click_list_div[j][0]]][preno]=1
                    category_adj[cat_dict[news_category[click_list_div[j][0]]]][cat_dict[news_category[preid]]]=1
                    category_adj[cat_dict[news_category[preid]]][cat_dict[news_category[click_list_div[j][0]]]]=1
                news_c_adj.append((cat_dict[news_category[click_list_div[j][0]]],news_dict[click_list_div[j][0]]))
                preno=news_dict[click_list_div[j][0]]
                preid=click_list_div[j][0]
                if(len(click)>=candidate_size+real_num):
                    break
                
        preno=-1
        preid=""
        for j in range(len(click_list)):
            if(click_list_div[j][-1]=='0' and click_list_div[j][0] in news_id):
                if (news_category[click_list_div[j][0]] not in cat_dict.keys()):
                    cat_dict[news_category[click_list_div[j][0]]]=len(cat_dict)
                # print(cat_dict)
                if (click_list_div[j][0] not in news_dict.keys()):
                    if (len(news_id[click_list_div[j][0]])<max_length):
                        news_id[click_list_div[j][0]].extend([0]*(max_length-len(news_id[click_list_div[j][0]])))
                    else:
                        news_id[click_list_div[j][0]]=news_id[click_list_div[j][0]][0:max_length]
                    news_list.append(news_id[click_list_div[j][0]])
                    news_dict[click_list_div[j][0]]=len(news_list)-1
                    clickrefuse.append(len(news_list)-1)
                else:
                    clickrefuse.append(news_dict[click_list_div[j][0]])
                news_list_used[news_dict[click_list_div[j][0]]].extend([batch_count%batch_size])
                if(preno!=-1 and len_click_id-j>3):
                    news_adj[preno][news_dict[click_list_div[j][0]]]=1
                    news_adj[news_dict[click_list_div[j][0]]][preno]=1    
                    category_adj[cat_dict[news_category[click_list_div[j][0]]]][cat_dict[news_category[preid]]]=1
                    category_adj[cat_dict[news_category[preid]]][cat_dict[news_category[click_list_div[j][0]]]]=1
                news_c_adj.append((cat_dict[news_category[click_list_div[j][0]]],news_dict[click_list_div[j][0]]))
                preno=news_dict[click_list_div[j][0]]
                preid=click_list_div[j][0]
                if(len(clickrefuse)>=refuse_num):
                    break
        
                
        batch_count+=1
        candidate=click[-real_num:]
        # candidate_entity=click_entity[-real_num:]
        # print(candidate)
        click=click[0:-real_num]
        if(len(click)<click_size):
            click.extend([-1]*(click_size-len(click)))
        if(len(clickrefuse)<refuse_num):
            clickrefuse.extend([-1]*(refuse_num-len(clickrefuse)))
        click_batch.append(click)
        clickrefuse_batch.append(clickrefuse)
        # click_entity_batch.append(click_entity)
        candidate_batch.append(candidate)
        if(batch_count==batch_size):
            # 先将新闻列表填满，再随机抽取候选新�?
            upno=len(news_name_list)-1
            while(len(news_list)<news_list_length):
                randno=random.randint(0,upno)
                if(news_name_list[randno] not in news_dict):
                    if (len(news_id[news_name_list[randno]])<=max_length):
                        news_id[news_name_list[randno]].extend([0]*(max_length-len(news_id[news_name_list[randno]])))
                    else:
                        news_id[news_name_list[randno]]=news_id[news_name_list[randno]][0:max_length]
                    news_list.append(news_id[news_name_list[randno]])
                    news_dict[news_name_list[randno]]=len(news_list)-1
            # 填充候选新�?
            for k in range(batch_size):
                while(len(candidate_batch[k])<candidate_size):
                    randno=random.randint(0,news_list_length-1)
                    # print(randno)
                    if(k not in news_list_used[randno]):
                        candidate_batch[k].extend([randno])
                        
            yield np.array(news_adj),np.array(news_list),np.array(click_batch),np.array(clickrefuse_batch),np.array(candidate_batch),news_c_adj,np.array(category_adj)
    return

def batch_iter_adj(users,news_id,news,news_category,category,category_num=50,batch_size=64,max_length=100,click_size=30,candidate_size=10,real_num=3,refuse_num=5):
    batch_count=0
    click_batch=[]
    clickrefuse_batch=[]
    candidate_batch=[]
    news_dict=dict() # 用新闻id得到在list的第几个
    news_list=[] # 存新闻内
    news_list_length=(click_size+real_num+refuse_num)*batch_size+1
    # 标记新闻属于哪个用户，使得候选新闻不出现历史新闻
    news_list_used=[[] for ii in range(news_list_length)]
    #news_adj=[[0 for ii in range(news_list_length)]for jj in range(news_list_length)]# [[0]*news_list_length]*news_list_length
    #news_c_adj=[]# [[0 for ii in range(news_list_length)]for jj in range(news_list_length)]
    #category_adj=[[0 for ii in range(category_num)]for jj in range(category_num)]
    news_name_list=list(news_id)
    for i in range(len(users)):
        if(batch_count==batch_size):
            batch_count=0
            click_batch=[]
            clickrefuse_batch=[]
            candidate_batch=[]
            # click_entity_batch=[]
            # candidate_entity_batch=[]
            news_list=[]
            news_dict=dict()
            news_list_used=[[] for ii in range(news_list_length)]
            news_adj=[[0 for ii in range(news_list_length)]for jj in range(news_list_length)]
            news_c_adj=[]# [[0 for ii in range(news_list_length)]for jj in range(news_list_length)]
            category_adj=[[0 for ii in range(category_num)]for jj in range(category_num)]
            
        click_id=str(users.history.iloc[i]).split(" ")
        refuse_id=str(users.click.iloc[i]).split(" ")
        click=[]
        clickrefuse=[]
        # click_entity=[]
        preno=-1
        preid=""
        len_click_id=len(click_id) if len(click_id)<=click_size+real_num else click_size+real_num
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
                # click.append(news_id[click_id[j]])
                else:
                    click.append(news_dict[click_id[j]])
                # print(news_dict[click_id[j]])
                # print(news_list_used)
                news_list_used[news_dict[click_id[j]]].extend([batch_count%batch_size])
                # click_entity.append(news_entity[click_id[j]])
                # print(preno,len(category))
                if(preno!=-1 and len_click_id-j>3):
                    news_adj[preno][news_dict[click_id[j]]]=1
                    news_adj[news_dict[click_id[j]]][preno]=1
                    # print(news_category[click_id[j]],news_category[preid])
                    # print(len(category_adj))
                    category_adj[news_category[click_id[j]]][news_category[preid]]=1
                    category_adj[news_category[preid]][news_category[click_id[j]]]=1
                news_c_adj.append((news_category[click_id[j]],news_dict[click_id[j]]))
                # news_c_adj[news_dict[click_id[j]]][news_category[click_id[j]]]=1
                    # print(news_adj)
                preno=news_dict[click_id[j]]
                preid=click_id[j]
            # else:
                # j-=1
        preno=-1
        preid=""
        for j in range(len(refuse_id)):
            # print(refuse_id[j])
            if(refuse_id[j][-1]=='0' and refuse_id[j][:-2] in news_id):
                # print(refuse_id[j][:-2])
                if (refuse_id[j][:-2] not in news_dict.keys()):
                    if (len(news_id[refuse_id[j][:-2]])<max_length):
                        news_id[refuse_id[j][:-2]].extend([0]*(max_length-len(news_id[refuse_id[j][:-2]])))
                    else:
                        news_id[refuse_id[j][:-2]]=news_id[refuse_id[j][:-2]][0:max_length]
                    news_list.append(news_id[refuse_id[j][:-2]])
                    news_dict[refuse_id[j][:-2]]=len(news_list)-1
                    clickrefuse.append(len(news_list)-1)
                # click.append(news_id[click_id[j]])
                else:
                    clickrefuse.append(news_dict[refuse_id[j][:-2]])
                # print(news_dict[click_id[j]])
                # print(news_list_used)
                news_list_used[news_dict[refuse_id[j][:-2]]].extend([batch_count%batch_size])
                # click_entity.append(news_entity[refuse_id[j][:-2]])
                if(preno!=-1 and len_click_id-j>3):
                    news_adj[preno][news_dict[refuse_id[j][:-2]]]=1
                    news_adj[news_dict[refuse_id[j][:-2]]][preno]=1    
                    category_adj[news_category[refuse_id[j][:-2]]][news_category[preid]]=1
                    category_adj[news_category[preid]][news_category[refuse_id[j][:-2]]]=1
                news_c_adj.append((news_category[refuse_id[j][:-2]],news_dict[refuse_id[j][:-2]]))
                preno=news_dict[refuse_id[j][:-2]]
                preid=refuse_id[j][:-2]
                if(len(clickrefuse)>=refuse_num):
                    break
        batch_count+=1
        candidate=click[-real_num:]
        click=click[0:-real_num]
        if(len(click)<click_size):
            click.extend([-1]*(click_size-len(click)))
        if(len(clickrefuse)<refuse_num):
            clickrefuse.extend([-1]*(refuse_num-len(clickrefuse)))
        click_batch.append(click)
        clickrefuse_batch.append(clickrefuse)
        # click_entity_batch.append(click_entity)
        candidate_batch.append(candidate)
        if(batch_count==batch_size):
            # 先将新闻列表填满，再随机抽取候选新
            upno=len(news_name_list)-1
            while(len(news_list)<news_list_length):
                randno=random.randint(0,upno)
                if(news_name_list[randno] not in news_dict):
                    if (len(news_id[news_name_list[randno]])<=max_length):
                        news_id[news_name_list[randno]].extend([0]*(max_length-len(news_id[news_name_list[randno]])))
                    else:
                        news_id[news_name_list[randno]]=news_id[news_name_list[randno]][0:max_length]
                    news_list.append(news_id[news_name_list[randno]])
                    news_dict[news_name_list[randno]]=len(news_list)-1
            # 填充候选新
            for k in range(batch_size):
                while(len(candidate_batch[k])<candidate_size):
                    randno=random.randint(0,news_list_length-1)
                    # print(randno)
                    if(k not in news_list_used[randno]):
                        candidate_batch[k].extend([randno])
            R = sp.dok_matrix((batch_size, len(news_list)), dtype=np.float32)
            for ii in range(batch_size):
                for jj in range(len(click[ii])):
                    if(click_batch[ii][jj]!=-1):
                        R[ii,click_batch[ii][jj]]=1.0
            yield R.to_csr(),np.array(news_list),np.array(click_batch),np.array(clickrefuse_batch),np.array(candidate_batch),news_c_adj,np.array(category_adj)
            #click_entity_batch,candidate_batch,candidate_entity_batch,candidate_batch[0]
    return

'''
class ENConfig(object):
    embedding_dim = 128  # 词向量维
    seq_length = 50  # 序列长度
    num_classes = 10  # 类别
    num_filters = 256  # 卷积核数
    kernel_size = 3  # 卷积核尺
    vocab_size = 50000  # 词汇表大
    attention_size = 64
    
    hidden_dim = 256  # 全连接层神经

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 2e-4  # 学习

    batch_size = 4  # 每批训练大小
    num_epochs = 2    # 总迭代轮

    print_per_batch = 30  # 每多少轮输出一次结
    save_per_batch = 10  # 每多少轮存入tensorboard
    
    # For additive attention
    query_vector_dim = 128
    num_words_title = seq_length
    
    candidate_len = 5
    click_len = 10
    real_num=2
    accept_num=5
    refuse_len = 5
    news_list_length=(click_len+real_num)*batch_size+1
    
    num_attention_heads = 16
    
    model='cnn' 
    rnn_num=3

    category_num = 40
    category_embedding_dim = 64


users=pd.read_csv("E:/2019sitp/data/endata/train_plus.tsv",sep='\t',index_col=False)
news=pd.read_csv("E:/2019sitp/data/endata/news_parsed.tsv",sep='\t',index_col=False)
# test=pd.read_csv("D:/2019sitp/data/endata/test.tsv")
news_content,content,news_entity,entity=news_process(news)
config=ENConfig()
for epoch in range(config.num_epochs):
    print('Epoch:', epoch + 1)
    batch_train = batch_iter(users,news_content,content,news_entity,entity,
                                    batch_size=config.batch_size,max_length=config.num_words_title,
                                    candidate_size=config.candidate_len,click_size=config.click_len,real_num=config.real_num,refuse_num=config.refuse_len)
    i=0
    for news_adj,news_list,click,refuse,candidate,news_c_adj,category_adj in batch_train:
        # print(candidate,click)
        np.array(news_list)
        i+=1
        print(i)
        print(click[0],candidate[0],news_list,news_list[0])
        if(i>=5):
            break
'''