import argparse
import torch
# import torchtext.legacy.data as data
# from torchtext.vocab import Vectors
import pandas as pd
import model
import model0
import train
import data_pro

parser = argparse.ArgumentParser(description='TextCNN text classifier')
parser.add_argument('-lr', type=float, default=0.0002, help='initial learning rate [default: 0.0002]')
parser.add_argument('-num-epochs', type=int, default=10, help='number of epochs for train [default: 10]')
parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 64]')
parser.add_argument('-click-size', type=int, default=30, help='history size for a user [default: 30]')
parser.add_argument('-candidate-size', type=int, default=10, help='candidate size for a user [default: 10]')
parser.add_argument('-real-size', type=int, default=3, help='positive news size for a user [default: 3]')
parser.add_argument('-refuse-size', type=int, default=3, help='the news refused to click by a user [default: 3]')
parser.add_argument('-num-words-title', type=int, default=100, help='number of words for a news [default: 300]')
parser.add_argument('-query-vector-dim', type=int, default=128, help='number of query vector\'s dimension [default: 128]')
parser.add_argument('-num-attention-heads', type=int, default=16, help='number of num attention heads [default: 16]')
parser.add_argument('-vocab-size', type=int, default=250000, help='size of vocabulary [default: 50000]')
parser.add_argument('-category-num', type=int, default=500, help='size of category [default: 200]')

parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stopping', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')

parser.add_argument('-test0', type=bool, default=False, help='the control group test')
parser.add_argument('-testzhihu', type=bool, default=False, help='the dataset testzhihu')

# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-refuse-rate', type=float, default=0.5, help='the rate of refuse feature [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embedding-dim', type=int, default=198, help='number of embedding dimension [default: 196]')
parser.add_argument('-num-filters', type=int, default=198, help='number of each size of filter')
parser.add_argument('-embedding-mid-dim', type=int, default=48, help='number of middle embedding dimension [default: 196]')
parser.add_argument('-kernel-size', type=int, default=3,help='comma-separated filter sizes to use for convolution')

parser.add_argument('-static', type=bool, default=False, help='whether to use static pre-trained word vectors')
parser.add_argument('-non-static', type=bool, default=False, help='whether to fine-tune static pre-trained word vectors')
parser.add_argument('-multichannel', type=bool, default=False, help='whether to use 2 channel of word vectors')
parser.add_argument('-pretrained-name', type=str, default='sgns.zhihu.word',
                    help='filename of pre-trained word vectors')
parser.add_argument('-pretrained-path', type=str, default='pretrained', help='path of pre-trained word vectors')

# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')

# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
args = parser.parse_args()

'''
def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    #print("vector",vectors)
    return vectors


def load_dataset(text_field, label_field, args, **kwargs):
    train_dataset, dev_dataset = dataset.get_dataset('data', text_field, label_field)
    
    if args.static and args.pretrained_name and args.pretrained_path:
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
        text_field.build_vocab(train_dataset, dev_dataset, vectors=vectors)
    else:
        text_field.build_vocab(train_dataset, dev_dataset)
    label_field.build_vocab(train_dataset, dev_dataset)
    train_iter, dev_iter = data.Iterator.splits(
        (train_dataset, dev_dataset),
        batch_sizes=(args.batch_size, len(dev_dataset)),
        sort_key=lambda x: len(x.text),
        **kwargs)
    return train_iter, dev_iter


print('Loading data...')
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter = load_dataset(text_field, label_field, args, device=-1, repeat=False, shuffle=True)
print("*",train_iter,dev_iter)
args.vocabulary_size = len(text_field.vocab)
if args.static:
    args.embedding_dim = text_field.vocab.vectors.size()[-1]
    args.vectors = text_field.vocab.vectors
if args.multichannel:
    args.static = True
    args.non_static = True
args.class_num = len(label_field.vocab)
'''
args.cuda = args.device != -1 and torch.cuda.is_available()

# args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]
print('Loading data...')
users=pd.read_csv("../data/endata/train_plus.tsv",sep='\t',index_col=False) if args.testzhihu is False else pd.read_csv("../data/zhihu/zhihu20M.txt", header=None, usecols=[0,1,2],sep='\t', index_col=False,names=["userid","num","news"]) 
test=pd.read_csv("../data/endata/test_plus_min.tsv",sep='\t',index_col=False) if args.testzhihu is False else pd.read_csv("../data/zhihu/test_min.txt", header=None, index_col=False,names=["userid","num","news"]) 
news=pd.read_csv("../data/endata/news_parsed.tsv",sep='\t',index_col=False) if args.testzhihu is False else pd.read_csv("../data/zhihu/answer_infos_min.txt", header=None, index_col=False,names=["newsid","content","category"])

if args.testzhihu:
    news.dropna(inplace=True)
    news.reset_index(drop=True, inplace=True) 
    args.save_dir="snapshotzhihu"
    # args.category_num=200

# config=ENConfig()
args.news_list_length=(args.click_size+args.real_size)*args.batch_size+1
print('Parameters:')
for attr, value in sorted(args.__dict__.items()):
    if attr in {'vectors'}:
        continue
    print('\t{}={}'.format(attr.upper(), value))
    
news_content,content,news_entity,entity=data_pro.news_process(news)  if args.testzhihu is False else data_pro.news_process_zh(news)

model = (model0.MODEL0(args) if args.test0 else model.MODEL(args))
if args.test0:
    args.save_dir="snapshot0"
if args.snapshot:
    print('\nLoading model from {}...\n'.format(args.snapshot))
    model.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    model = model.cuda()
    
batch_iter= data_pro.batch_iter if args.testzhihu is False else data_pro.batch_iter_zh
try:
    batch_train = batch_iter(users,news_content,content,news_entity,entity,
                                    category_num=args.category_num,batch_size=args.batch_size,max_length=args.query_vector_dim,
                                    candidate_size=args.candidate_size,click_size=args.click_size,real_num=args.real_size,refuse_num=args.refuse_size)
    train.train(batch_train, model, args, test,news_content,content,news_entity,entity)
except KeyboardInterrupt:
    print('Exiting from training early')
