import os
import sys
import torch
import torch.nn.functional as F
import data_pro

def train(train_iter, model, args , test, news_content, content, news_entity, entity):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.num_epochs + 1):
        for news_adj,news_list,click,refuse,candidate, news_c_adj ,category_adj in train_iter:
            news_adj=torch.from_numpy(news_adj)
            news_list=torch.from_numpy(news_list)
            click=torch.from_numpy(click)
            refuse=torch.from_numpy(refuse)
            candidate=torch.from_numpy(candidate)
            # print(torch.max(news_list))
            category_adj = torch.from_numpy(category_adj)
            # print(category_adj)
            if args.cuda and args.test0 == False:
                news_adj,news_list,click,refuse,candidate, category_adj = news_adj.cuda(),news_list.cuda(),click.cuda(),refuse.cuda(),candidate.cuda(), category_adj.cuda()
            if args.cuda and args.test0 :
                news_list,click,refuse,candidate = news_list.cuda(),click.cuda(),refuse.cuda(),candidate.cuda()
                # feature, target = feature.cuda(), target.cuda()
            possible,loss = model(news_adj, news_list, click, refuse, candidate, news_c_adj ,category_adj) 
            # loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            if steps % args.log_interval == 0 and steps!=0:
                train_acc=model.ACC(possible)
                corrects=model.MRR(possible)
                auc=model.AUC(possible)
                # corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
                # train_acc = 100.0 * corrects / args.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  dcg: {:.4f} mrr: {:.4f} auc: {:.4f} '.format(steps,
                                                                             loss.item(),
                                                                             train_acc.item(),
                                                                             corrects.item(),auc))
            if steps % args.test_interval == 0 and steps!=0:
                debug=False
                if steps==200:
                    debug = True
                dev_acc = eval(model, args , test, news_content, content, news_entity, entity, debug)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stopping:
                        print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
                        raise KeyboardInterrupt
            steps += 1



def eval(model, args, test, news_content, content, news_entity, entity, debug=False):
    batch_iter= data_pro.batch_iter if args.testzhihu is False else data_pro.batch_iter_zh
    batch_test = batch_iter(test,news_content,content,news_entity,entity,
                                    category_num=args.category_num,batch_size=args.batch_size,max_length=args.query_vector_dim,
                                    candidate_size=args.candidate_size,click_size=args.click_size,real_num=args.real_size,refuse_num=args.refuse_size)
    model.eval()
    dcgs, mrrs, avg_loss, aucs = 0, 0, 0, 0
    size=0
    for news_adj,news_list,click,refuse, candidate, news_c_adj ,category_adj in batch_test:
        # print(size)
        news_adj=torch.from_numpy(news_adj)
        news_list=torch.from_numpy(news_list)
        click=torch.from_numpy(click)
        refuse=torch.from_numpy(refuse)
        candidate=torch.from_numpy(candidate)
        # news_c_adj = torch.from_numpy(news_c_adj)
        category_adj = torch.from_numpy(category_adj)
        # print(news_list)
        # print(news_c_adj)
        # prinT(category_adj)
        if args.cuda:
            news_adj,news_list,click,refuse,candidate,category_adj=news_adj.cuda(),news_list.cuda(),click.cuda(),refuse.cuda(),candidate.cuda() ,category_adj.cuda()
        possible,loss = model(news_adj, news_list, click, refuse, candidate, news_c_adj ,category_adj) 
        avg_loss += loss
        mrrs += model.MRR(possible,debug).item()
        dcgs += model.ACC(possible,debug).item()
        aucs += model.AUC(possible,debug)
        size += 1
        if(size>30):
            break
        
    avg_loss /= size
    mrrs = 100.0 * mrrs / size
    dcgs = 100.0 * dcgs / size
    aucs = 100.0 * aucs / size
    print(
        '\nEvaluation - loss: {:.6f}  dcg: {:.4f} mrr: {:.4f} auc: {:.4f}'.format(avg_loss,
                                                                 dcgs,
                                                                 mrrs,
                                                                 aucs))
    '''
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    '''
    return dcgs


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)

def prinT(content):
    print("")
    for i in range(0,len(content)):
        for j in range(0,len(content[i])):
            if(content[i,j]==1):
                print(i,j)