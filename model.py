import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl.function as fn
import dgl
import numpy as np

class MODEL(torch.nn.Module):
    """
    TANR network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self, config, pretrained_word_embedding=None):
        super(MODEL, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = UserEncoder(config)
        self.category_encoder = CategoryEncoder(config)
        self.click_predictor = DotProductClickPredictor()
        self.gcn = GCN(self.config.num_filters,self.config.embedding_mid_dim,self.config.cuda)
        self.gcn2 = GCN(self.config.num_filters,self.config.embedding_mid_dim,self.config.cuda)
        self.rgcn0=heteroRGCNLayer(self.config.embedding_dim,self.config.embedding_dim,["news","1","category"],self.config.cuda)
        self.rgcn=heteroRGCNLayer(self.config.embedding_mid_dim,self.config.embedding_mid_dim,["category","1","news"],self.config.cuda)


    def forward(self,news_adj, news_list, clicked_news, refuse_news, candidate_news, news_c_adj ,category_adj):
        """
        Args:
            candidate_news:
                [
                    {
                        "category": batch_size,
                        "title": batch_size * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "category": batch_size,
                        "title": batch_size * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
            click_probability: batch_size, 1 + K
            topic_classification_loss: 0-dim tensor
        """
        # news_list_length,num_filters
        news_feature=self.news_encoder(news_list)
        if self.config.cuda:
            news_feature=news_feature.cuda()
        # normalize
        news_adj_normal=self.normalize(news_adj)
        # print(news_feature.device)
        news_feature_gcn=self.gcn(news_adj_normal,news_feature)
        # news_feature_gcn=news_feature
        # print(news_feature_gcn.size())        
        padding=torch.zeros((1,self.config.embedding_mid_dim),dtype=torch.float)
        if self.config.cuda:
            padding=padding.cuda()
        #---------------------------------category----------------------------------------------------------
        categories=np.zeros((self.config.category_num),dtype= np.int)
        for i in range(1,self.config.category_num):
            categories[i]=i
        categories=torch.from_numpy(categories)
        if self.config.cuda:
            categories=categories.cuda()
        cat_feature=self.category_encoder(categories)
        news_c_adj.append((self.config.category_num-1,news_feature_gcn.size()[0]-1))
        '''
        #---------------------------------news and category------------------------------------------------
        len_edges=len(news_c_adj)
        c_news_adj=[]
        for i in range(0,len_edges):
            c_news_adj.append((news_c_adj[i][1],news_c_adj[i][0]))
        device="cuda:"+str(self.config.device)
        ncgraph0=dgl.heterograph({('news','1','category'):c_news_adj},device=device)
        cat_feature=self.rgcn0(ncgraph0,{"news":news_feature,"category":cat_feature})
        '''
        #---------------------------------category to category---------------------------------------------
        category_adj_normal=self.normalize(category_adj)
        # print(category_adj_normal.size(),cat_feature.size())
        cat_feature_gcn=self.gcn2(category_adj_normal,cat_feature)     
        #---------------------------------news and category------------------------------------------
        # normalize
        # print(cat_feature_gcn.size())
        # print(news_c_adj)
        device="cuda:"+str(self.config.device)
        ncgraph=dgl.heterograph({('category','1','news'):news_c_adj},device=device)
        news_feature_gcn=self.rgcn(ncgraph,{"category":cat_feature_gcn,"news":news_feature_gcn})
        #-----------------------------------------------------------------------------------------------
        
        click_news_vector_one=padding
        for j in clicked_news[0]:
            if(j !=-1):
                # print(click_news_vector_one.size(),news_feature_gcn[j].size())
                click_news_vector_one=torch.cat((click_news_vector_one,news_feature_gcn[j].unsqueeze(0)),dim=0)
            else:
                click_news_vector_one=torch.cat((click_news_vector_one,padding),dim=0)
        click_news_vector=click_news_vector_one[1:].unsqueeze(0)
        for i in range(1,self.config.batch_size):
            click_news_vector_one=padding
            for j in clicked_news[i]:
                if(j !=-1):
                    click_news_vector_one=torch.cat((click_news_vector_one,news_feature_gcn[j].unsqueeze(0)),dim=0)
                else:
                    click_news_vector_one=torch.cat((click_news_vector_one,padding),dim=0)
            # print(click_news_vector_one.size())
            click_news_vector=torch.cat((click_news_vector,click_news_vector_one[1:].unsqueeze(0)),dim=0)
        user_vector = self.user_encoder(click_news_vector)
        #-----------------------------------------------------------------------------------------------
        refuse_news_vector_one=padding
        for j in refuse_news[0]:
            if(j !=-1):
                # print(refuse_news_vector_one.size(),news_feature_gcn[j].size())
                refuse_news_vector_one=torch.cat((refuse_news_vector_one,news_feature_gcn[j].unsqueeze(0)),dim=0)
            else:
                refuse_news_vector_one=torch.cat((refuse_news_vector_one,padding),dim=0)
        refuse_news_vector=refuse_news_vector_one[1:].unsqueeze(0)
        for i in range(1,self.config.batch_size):
            refuse_news_vector_one=padding
            for j in refuse_news[i]:
                if(j !=-1):
                    refuse_news_vector_one=torch.cat((refuse_news_vector_one,news_feature_gcn[j].unsqueeze(0)),dim=0)
                else:
                    refuse_news_vector_one=torch.cat((refuse_news_vector_one,padding),dim=0)
            # print(refuse_news_vector_one.size())
            refuse_news_vector=torch.cat((refuse_news_vector,refuse_news_vector_one[1:].unsqueeze(0)),dim=0)
        user_refuse_vector = self.user_encoder(refuse_news_vector)
        #-------------------------------------------------------------------------------------------------
        candidate_news_vector_one=padding
        for j in candidate_news[0]:
            if(j !=-1):
                # print(candidate_news_vector_one.size(),news_feature_gcn[j].size())
                candidate_news_vector_one=torch.cat((candidate_news_vector_one,news_feature_gcn[j].unsqueeze(0)),dim=0)
            else:
                candidate_news_vector_one=torch.cat((candidate_news_vector_one,padding),dim=0)
        candidate_news_vector=candidate_news_vector_one[1:].unsqueeze(0)
        for i in range(1,self.config.batch_size):
            candidate_news_vector_one=padding
            for j in candidate_news[i]:
                if(j !=-1):
                    candidate_news_vector_one=torch.cat((candidate_news_vector_one,news_feature_gcn[j].unsqueeze(0)),dim=0)
                else:
                    candidate_news_vector_one=torch.cat((candidate_news_vector_one,padding),dim=0)
            # print(candidate_news_vector_one.size())
            candidate_news_vector=torch.cat((candidate_news_vector,candidate_news_vector_one[1:].unsqueeze(0)),dim=0)
        # print(candidate_news_vector.size())

        '''
        # batch_size, 1 + K, num_filters
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news], dim=1)
        # batch_size, num_clicked_news_a_user, num_filters
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)
        '''
        # batch_size, num_filters
        # user_vector = self.user_encoder(clicked_news_vector)
        # batch_size, 1 + K
        click_probability = torch.softmax(self.click_predictor(candidate_news_vector,
                                                 user_vector),dim=1)
        # refuse_probability = torch.softmax(self.click_predictor(candidate_news_vector,
        #                                         user_refuse_vector),dim=1)
        last_probability = click_probability# -self.config.refuse_rate*refuse_probability
        # print(click_probability,torch.sum(last_probability[:,0:self.config.real_size],dim=1))
        # Sloss =torch.mean(torch.sum(last_probability[:,0:self.config.real_size],dim=1),dim=0)
        '''
        loss=last_probability[:][0]
        for i in range(1,self.config.real_size):
            loss=loss.mul(last_probability[:][i])
        '''
        crossentropyloss=nn.CrossEntropyLoss()
        loss=0
        targets=torch.zeros((self.config.batch_size),dtype=torch.long).cuda()
        # torch.cat((click_probability[:,0],click_probability[:,self.config.real_size:]),1)
        for i in range(0,self.config.real_size):
            loss+=crossentropyloss(torch.cat((click_probability[:,i:i+1],click_probability[:,self.config.real_size:]),1),targets)
        # [1,2,3] [1,2,3]
        # [4,4,4] [5,5,5]
        """
        # batch_size * (1 + K + num_clicked_news_a_user), num_categories
        y_pred = self.topic_predictor(
            torch.cat((candidate_news_vector, clicked_news_vector),
                      dim=1).view(-1, self.config.num_filters))
        # batch_size * (1 + K + num_clicked_news_a_user)
        y = torch.stack([x['category'] for x in candidate_news + clicked_news],
                        dim=1).flatten().to(device)
        class_weight = torch.ones(self.config.num_categories).to(device)
        class_weight[0] = 0
        criterion = nn.CrossEntropyLoss(weight=class_weight)
        topic_classification_loss = criterion(y_pred, y)
        """
        return last_probability,loss #, topic_classification_loss
    
    def MRR(self,click_probability,debug=False):
        rank=torch.ones((self.config.batch_size,1),dtype=torch.int)
        real_max=torch.max(click_probability[:,0:self.config.real_size],dim=1)[0]
        # print("mrr",click_probability.size(),real_max.size())
        for i in range(self.config.batch_size):
            for j in range(self.config.real_size,self.config.candidate_size):
                if(click_probability[i,j].item()>=real_max[i].item()):
                    rank[i]+=1
            if i==0 and debug:
                print("MRR")
                print(click_probability[i],rank[i])
        mrr=torch.sum(1.0/rank)/len(rank)
        return mrr
    
    def ACC(self,click_probability,debug=False):
        acc=torch.zeros((self.config.batch_size,1))
        for i in range(self.config.batch_size):
            rank={}
            for j in range(0,self.config.candidate_size):
                rank[j]=click_probability[i,j]
            rank,_=zip(*sorted(rank.items(), key = lambda kv:(-kv[1], -kv[0])))
            for j in range(self.config.candidate_size):
                if(rank[j]<self.config.real_size):
                    acc[i]+=(2-1)/math.log(j+2,2)
            '''
            if debug and i==0:
                print("ACC")
                print(click_probability[i],rank,acc[i])
            '''
                #if(rank[j]<config.real_size and j<config.accept_num):
                    #acc[i]+=1
        idcg=0.0
        for i in range(self.config.real_size):
            idcg+=(2-1)/math.log(i+2,2)
        return torch.mean(acc)/(idcg)
    
    def LOSS(self,click_probability,debug=False):
        n=self.config.real_size*(self.config.candidate_size-self.config.real_size)
        # score_tot=torch.zeros((self.config.batch_size,n),requires_grad=True)
        y_p=click_probability[:,0:self.config.real_size].repeat(1,self.config.candidate_size-self.config.real_size)
        y_n=click_probability[:,self.config.real_size:self.config.candidate_size].view(self.config.batch_size,self.config.candidate_size-self.config.real_size,1)
        y_n=y_n.expand(self.config.batch_size,self.config.candidate_size-self.config.real_size,self.config.real_size)
        # print(y_n.size())
        y_n=y_n.contiguous().view(self.config.batch_size,n)
        # print(y_p,y_n)
        # print(y_p[0],y_n[0])       
        score_tot=y_p-y_n# torch.gt(y_p,y_n)
        '''
        for i in range(self.config.batch_size):
            # score=0
            for j in range(self.config.real_size):
                for k in range(self.config.real_size,self.config.candidate_size):
                    # if(click_probability[i,j]>click_probability[i,k]):
                    score_tot[i,j*k+k]=torch.gt(click_probability[i,j],click_probability[i,k]).clone()
            # score_tot[i]=(score/n)
            if debug and i==0:
                print(click_probability[i],score_tot[i])
        '''
        # score_tot= torch.where(torch.isnan(score_tot), torch.full_like(score_tot, 0), score_tot)# score_tot**0.2
        score_tot= torch.where(score_tot>0.5,torch.full_like(score_tot, 0.5), score_tot)
        
        return torch.mean(score_tot)
    
    def AUC(self,click_probability,debug=False):
        n=self.config.real_size*(self.config.candidate_size-self.config.real_size)
        score_tot=torch.zeros((self.config.batch_size,n))
        for i in range(self.config.batch_size):
            score=0
            for j in range(self.config.real_size):
                for k in range(self.config.real_size,self.config.candidate_size):
                    # if(click_probability[i,j]>click_probability[i,k]):
                    score+=torch.gt(click_probability[i,j],click_probability[i,k])
            score_tot[i]=(score/n)
            '''
            if debug and i==0:
                print("AUC",click_probability[i],score_tot[i])
            '''
        return torch.mean(score_tot)
    
    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, num_filters
        """
        # batch_size, num_filters
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters
        Returns:
            (shape) batch_size, num_filters
        """
        # batch_size, num_filters
        return self.user_encoder(clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size, word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        # candidate_size
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)
        
    def normalize(self, A , symmetric=True):
        # A = A+I
        I = torch.eye(A.size(0))
        if self.config.cuda:
            I = I.cuda()
        A = A + I
        d = A.sum(1)
        if symmetric:
        #D = D^-1/2
            D = torch.diag(torch.pow(d , -0.5))
            return D.mm(A).mm(D)
        else :
            # D=D^-1
            D =torch.diag(torch.pow(d,-1))
            return D.mm(A)
"""        
class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        self.tot_news_num = args.click_size*args.batch_size
        
        class_num = args.class_num
        chanel_num = 1
        filter_num = args.filter_num
        filter_sizes = args.filter_sizes

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
            chanel_num += 1
        else:
            self.embedding2 = None
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x):
        if self.embedding2:
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            x = self.embedding(x)
            x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
"""

class CategoryEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_cat_embedding=None):
        super(CategoryEncoder, self).__init__()
        self.config = config
        if pretrained_cat_embedding is None:
            self.cat_embedding = nn.Embedding(config.category_num,
                                               config.embedding_dim,
                                               padding_idx=0)
        else:
            self.cat_embedding = nn.Embedding.from_pretrained(
                pretrained_cat_embedding, freeze=False, padding_idx=0)

    def forward(self, categories):
        return F.dropout(F.relu(self.cat_embedding(categories)),
                                           p=self.config.dropout,
                                           training=self.training)
        
class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding=None):
        super(NewsEncoder, self).__init__()
        self.config = config
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.vocab_size,
                                               config.embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
        assert config.kernel_size >= 1 and config.kernel_size % 2 == 1
        self.title_CNN = nn.Conv2d(
            1,
            config.num_filters,
            (config.kernel_size, config.embedding_dim),
            padding=(int((config.kernel_size - 1) / 2), 0))
        self.title_attention = AdditiveAttention(config.query_vector_dim,
                                                 config.num_filters)

    def forward(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, num_filters
        """
        # batch_size, num_words_title, word_embedding_dim
        
        title_vector = F.dropout(self.word_embedding(news),
                                 p=self.config.dropout,
                                 training=self.training)
        # batch_size, num_filters, num_words_title
        convoluted_title_vector = self.title_CNN(
            title_vector.unsqueeze(dim=1)).squeeze(dim=3)
        # batch_size, num_filters, num_words_title
        activated_title_vector = F.dropout(F.relu(convoluted_title_vector),
                                           p=self.config.dropout,
                                           training=self.training)
        # batch_size, num_filters
        weighted_title_vector = self.title_attention(
            activated_title_vector.transpose(1, 2))

        return weighted_title_vector
    
class GCN(nn.Module):
    def __init__(self , dim_in , dim_out, is_gpu=False):
        super(GCN,self).__init__()
        self.fc1 = nn.Linear(dim_in ,dim_out,bias=False)
        if is_gpu:
            self.fc1=self.fc1.cuda()
        '''
        self.fc2 = nn.Linear(dim_in,dim_in//4*3,bias=False)
        if is_gpu:
            self.fc2=self.fc2.cuda()
        
        self.fc3 = nn.Linear(dim_in,dim_out,bias=False)
        if is_gpu:
            self.fc3=self.fc3.cuda()
        '''

    def forward(self,A,X):
        '''
        计算三层gcn
        '''
        # print(self.A.device,X.device)
        # print(self.fc1.device)
        X = F.relu(self.fc1(A.mm(X)))
        # X = F.relu(self.fc2(A.mm(X)))
        return X # self.fc3(A.mm(X))

class heteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes,is_gpu=False):
        super(heteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
                name : (nn.Linear(in_size, out_size).cuda() if is_gpu else nn.Linear(in_size, out_size)) for name in etypes})
        
    def forward(self, G, feat_dict):
        funcs = {}
        # print(G.canonical_etypes)
        returnnode=""
        for srctype, etype, dsttype in G.canonical_etypes:
            returnnode=dsttype
            Wh = self.weight[etype](feat_dict[srctype])
            # print("wh",Wh)          
            G.nodes[srctype ] .data[ 'wh_%s'%etype] = Wh          
            funcs[etype] = (fn.copy_u( 'wh_%s' % etype,'m'), fn.mean( 'm', 'h'))
            # print("funcs",funcs)
            # print(G.nodes["news"])
        G.multi_update_all(funcs,'sum' )  
        return G.nodes[returnnode].data['h']
    
class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.additive_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.embedding_mid_dim)

    def forward(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters
        Returns:
            (shape) batch_size, num_filters
        """
        user_vector = self.additive_attention(clicked_news_vector)
        return user_vector
    
class AdditiveAttention(torch.nn.Module):
    """
    A general additive attention module.
    Originally for NAML.
    """
    def __init__(self,
                 query_vector_dim,
                 candidate_vector_dim,
                 writer=None,
                 tag=None,
                 names=None):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))
        # For tensorboard
        self.writer = writer
        self.tag = tag
        self.names = names
        self.local_step = 1

    def forward(self, candidate_vector):
        """
        Args:
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        # print(candidate_vector.size())
        # batch_size, candidate_size, query_vector_dim
        temp = torch.tanh(self.linear(candidate_vector))
        # batch_size, candidate_size
        candidate_weights = F.softmax(torch.matmul(
            temp, self.attention_query_vector),
                                      dim=1)
        if self.writer is not None:
            assert candidate_weights.size(1) == len(self.names)
            if self.local_step % 10 == 0:
                self.writer.add_scalars(
                    self.tag, {
                        x: y
                        for x, y in zip(self.names,
                                        candidate_weights.mean(dim=0))
                    }, self.local_step)
            self.local_step += 1
        # batch_size, candidate_vector_dim
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target
    
class DotProductClickPredictor(torch.nn.Module):
    def __init__(self):
        super(DotProductClickPredictor, self).__init__()

    def forward(self, candidate_news_vector, user_vector):
        """
        Args:
            candidate_news_vector: batch_size, candidate_size, X
            user_vector: batch_size, X
        Returns:
            (shape): batch_size
        """
        # batch_size, candidate_size
        probability = torch.bmm(candidate_news_vector,
                                user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        return probability