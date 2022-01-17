import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl.function as fn
import dgl
import numpy as np

class MODEL0(torch.nn.Module):
    """
    TANR network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self, config, pretrained_word_embedding=None):
        super(MODEL0, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = UserEncoder(config)
        self.category_encoder = CategoryEncoder(config)
        self.click_predictor = DotProductClickPredictor()
        

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
        padding=torch.zeros((1,self.config.embedding_dim),dtype=torch.float)
        if self.config.cuda:
            padding=padding.cuda()
        click_news_vector_one=padding
        for j in clicked_news[0]:
            if(j !=-1):
                # print(click_news_vector_one.size(),news_feature_gcn[j].size())
                click_news_vector_one=torch.cat((click_news_vector_one,news_feature[j].unsqueeze(0)),dim=0)
            else:
                click_news_vector_one=torch.cat((click_news_vector_one,padding),dim=0)
        click_news_vector=click_news_vector_one[1:].unsqueeze(0)
        for i in range(1,self.config.batch_size):
            click_news_vector_one=padding
            for j in clicked_news[i]:
                if(j !=-1):
                    click_news_vector_one=torch.cat((click_news_vector_one,news_feature[j].unsqueeze(0)),dim=0)
                else:
                    click_news_vector_one=torch.cat((click_news_vector_one,padding),dim=0)
            # print(click_news_vector_one.size())
            click_news_vector=torch.cat((click_news_vector,click_news_vector_one[1:].unsqueeze(0)),dim=0)
        user_vector = self.user_encoder(click_news_vector)
        #-------------------------------------------------------------------------------------------------
        candidate_news_vector_one=padding
        for j in candidate_news[0]:
            if(j !=-1):
                # print(candidate_news_vector_one.size(),news_feature_gcn[j].size())
                candidate_news_vector_one=torch.cat((candidate_news_vector_one,news_feature[j].unsqueeze(0)),dim=0)
            else:
                candidate_news_vector_one=torch.cat((candidate_news_vector_one,padding),dim=0)
        candidate_news_vector=candidate_news_vector_one[1:].unsqueeze(0)
        for i in range(1,self.config.batch_size):
            candidate_news_vector_one=padding
            for j in candidate_news[i]:
                if(j !=-1):
                    candidate_news_vector_one=torch.cat((candidate_news_vector_one,news_feature[j].unsqueeze(0)),dim=0)
                else:
                    candidate_news_vector_one=torch.cat((candidate_news_vector_one,padding),dim=0)
            candidate_news_vector=torch.cat((candidate_news_vector,candidate_news_vector_one[1:].unsqueeze(0)),dim=0)
        click_probability = torch.softmax(self.click_predictor(candidate_news_vector,
                                                 user_vector),dim=1)
        last_probability = click_probability# -self.config.refuse_rate*refuse_probability
        crossentropyloss=nn.CrossEntropyLoss()
        loss=0
        targets=torch.zeros((self.config.batch_size),dtype=torch.long).cuda()
        # torch.cat((click_probability[:,0],click_probability[:,self.config.real_size:]),1)
        for i in range(0,self.config.real_size):
            loss+=crossentropyloss(torch.cat((click_probability[:,i:i+1],click_probability[:,self.config.real_size:]),1),targets)
        # loss+=crossentropyloss(click_probability[:,:self.config.real_size],torch.zeros((self.config.batch_size,self.config.real_size),dtype=torch.long).cuda())
        # loss= torch.mean(torch.sum(click_probability[:,self.config.real_size:],dim=1)) #self.LOSS(last_probability)
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
                print("mrr",click_probability[i],rank[i])
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
            if debug and i==0:
                print("acc",click_probability[i],rank,acc[i])
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
        score_tot= torch.where(score_tot>0.4, torch.full_like(score_tot, 0.4), score_tot)
        return torch.mean(score_tot)
    
    def AUC(self,click_probability,debug=False):
        n=self.config.real_size*(self.config.candidate_size-self.config.real_size)
        score_tot=torch.zeros((self.config.batch_size,1))
        for i in range(self.config.batch_size):
            score=0
            for j in range(self.config.real_size):
                for k in range(self.config.real_size,self.config.candidate_size):
                    # if(click_probability[i,j]>click_probability[i,k]):
                    score+=torch.gt(click_probability[i,j],click_probability[i,k])
            score_tot[i]=(score/n)
            if debug and i==0:
                print("auc",click_probability[i],score_tot[i])
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
        
class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        # self.multihead_self_attention = MultiHeadSelfAttention(config.embedding_dim, config.num_attention_heads)
        self.additive_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.embedding_dim)

    def forward(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters
        Returns:
            (shape) batch_size, num_filters
        """
        # multihead_user_vector = self.multihead_self_attention(clicked_news_vector)
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
        
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_attention_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K=None, V=None, length=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.num_attention_heads,
                               self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.num_attention_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.num_attention_heads,
                               self.d_v).transpose(1, 2)

        if length is not None:
            maxlen = Q.size(1)
            attn_mask = torch.arange(maxlen).to(device).expand(
                batch_size, maxlen) < length.to(device).view(-1, 1)
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, maxlen,
                                                      maxlen)
            attn_mask = attn_mask.unsqueeze(1).repeat(1,
                                                      self.num_attention_heads,
                                                      1, 1)
        else:
            attn_mask = None

        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s,
                                                            attn_mask)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_attention_heads * self.d_v)
        return context