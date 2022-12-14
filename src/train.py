import os 
import sys
import os.path as osp
import random
import argparse
import numpy as np
import pprint
import torch
from tqdm import tqdm 
import itertools

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils.utils import mkdir_p, merge_args_yaml
from utils.prepare import *
from utils.modules import test, get_features
from utils.utils import save_models, load_models   
from models import metrics, focal_loss


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='CGFG')
    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/celeba.yml',
                        help='optional config file')
    parser.add_argument('--train', type=bool, default=True, help='if train model')
    args = parser.parse_args()
    return args


def get_loss(args):
    if args.loss == "focal_loss":
        criterion = focal_loss.FocalLoss(gamma=2)

    elif args.loss == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()

    return criterion


def get_margin(args):
    if args.metric == "add_margin":
        metric_fc = metrics.AddMarginProduct(args.fusion_final_dim, 
                                            args.num_classes, 
                                            s=30, m=0.35)
    elif args.metric == "arc_margin":
        metric_fc = metrics.ArcMarginProduct(args.fusion_final_dim, 
                                            args.num_classes, 
                                            s=30, m=0.5, 
                                            easy_margin=args.easy_margin)

    
    metric_fc.to(args.device)
    metric_fc = torch.nn.DataParallel(metric_fc, device_ids=args.gpu_id)
    return metric_fc


def get_optimizer(args, net, metric_fc, text_encoder, text_head):
    params = [{"params": metric_fc.parameters(), "lr" : args.lr_image_train, "weight_decay" : args.weight_decay}]
    params_en = [{"params": text_encoder.parameters(), "lr" : 5e-5}]
    params_head = [{"params": itertools.chain(text_head.parameters(), net.parameters()), "lr": args.lr_head}]

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params)
        optimizer_en = torch.optim.AdamW(params_en,  betas=(0.9, 0.999), weight_decay=0.01)
        optimizer_head = torch.optim.Adam(params_head, weight_decay=args.weight_decay) 

    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params)
        optimizer_head = torch.optim.Adam(params_head,  betas=(0.5, 0.999)) 
    return optimizer, optimizer_en, optimizer_head



def train(train_dl, model, net, metric_fc, text_encoder, text_head, criterion, 
        optimizer, optimizer_en, optimizer_head, scheduler, lr_scheduler_en, lr_scheduler_head, args):

    device = args.device
    net.train()
    metric_fc.train()
    if args.current_epoch < 11:
        text_encoder.train()
    else:
        text_encoder.eval()

    text_head.train()

    loop = tqdm(total=len(train_dl))
    total_loss = 0
    for step, data in enumerate(train_dl, 0):
        if args.using_BERT == True:
            imgs, words_emb, word_vector, sent_emb, sent_emb_org, keys, label = \
                    prepare_train_data_for_Bert(data, text_encoder, text_head)
            cap_lens = None 

        if args.using_BERT == False:
            imgs, words_emb, sent_emb, keys, label, cap_lens = \
                    prepare_train_data(data, text_encoder, text_head)
        
        # load cuda
        if args.using_BERT == True: word_vector = word_vector.to(device).requires_grad_()
        imgs = imgs.to(device).requires_grad_()
        words_emb = words_emb.to(device).requires_grad_()
        sent_emb = sent_emb.to(device).requires_grad_()
        sent_emb_org = sent_emb_org.to(device).requires_grad_()
        label = label.to(device)
        
        global_feats, local_feats = get_features(model, imgs)

        if args.fusion_type == "linear":
            output = net(global_feats, word_vector)

        elif args.fusion_type == "concat":
            output = net(global_feats, word_vector)

        elif args.fusion_type == "concat_attention":
            output = net(global_feats, word_vector)

        elif args.fusion_type == "paragraph_attention":
            output = net(global_feats, sent_emb_org)

        elif args.fusion_type == "cross_attention":
            output = net(local_feats, words_emb, global_feats, sent_emb)


        output = metric_fc(output, label)
        loss = criterion(output, label)
        total_loss += loss

        optimizer.zero_grad()
        if args.current_epoch < 11: optimizer_en.zero_grad()
        optimizer_head.zero_grad()
        loss.backward()

        optimizer.step()
        if args.current_epoch < 11: optimizer_en.step()
        optimizer_head.step()
        lr_scheduler_en.step()


        # update loop information
        loop.update(1)
        loop.set_description(f'Training Epoch [{args.current_epoch}/{args.max_epoch}]')
        loop.set_postfix()

    loop.close()
    scheduler.step()
    lr_scheduler_head.step(total_loss)

    print("learning rate: ", scheduler.get_last_lr())
    str_loss = " | loss {:0.4f}".format(total_loss.item() / step)
    print(str_loss)
    del global_feats, output
    


def main(args):
    # prepare dataloader, models, data
    train_dl, train_ds, valid_dl, valid_ds = prepare_dataloader(args, split="train", transform=None)
    test_dl, test_ds = prepare_dataloader(args, split="test", transform=None)

    if args.using_BERT == False:
        args.vocab_size = train_ds.n_words

    del train_ds, test_ds, valid_dl, valid_ds 

    text_encoder, text_head = prepare_text_encoder(args, test=False)
    
    model = prepare_model(args) #cuda + parallel + grd. false + eval
    net = prepare_fusion_net(args) #cuda + parallel
    metric_fc = get_margin(args) #cuda + parallel

    opt, opt_en, opt_head = get_optimizer(args, net, metric_fc, text_encoder, text_head)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, 
                                    step_size=args.lr_step, 
                                    gamma=args.gamma)

    
    lr_scheduler_head = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_head, mode="min", patience=args.patience, factor=args.factor)

    criterion = get_loss(args)
    
    # load from checkpoint
    strat_epoch = 1
    if args.resume_epoch!=1:
        print("loading checkpoint; epoch: ", args.resume_epoch)
        strat_epoch = args.resume_epoch+1
        net, metric_fc, opt = load_models(net, metric_fc, opt, args.resume_model_path)
    

    #pprint.pprint(args)
    print("Start Training")
    args.is_roc = False 
    for epoch in range(strat_epoch, args.max_epoch + 1):
        torch.cuda.empty_cache()
        args.current_epoch = epoch
        print('Reset scheduler')
        lr_scheduler_en = torch.optim.lr_scheduler.CosineAnnealingLR(opt_en, T_max=1000, eta_min=1e-5)

        train(train_dl, model, net, metric_fc, text_encoder, text_head, criterion, 
                    opt, opt_en, opt_head, scheduler, lr_scheduler_en, lr_scheduler_head, args)
    
        # save
        if (epoch > 8):
            if epoch % args.save_interval==0:
                save_models(net, metric_fc, opt, text_encoder, text_head, epoch, args)
            
            if ((args.do_test == True) and (epoch % args.test_interval == 0)):
                print("\nLet's test the model")
                test(test_dl, model, net, text_encoder, text_head, args)



if __name__ == "__main__":
    args = merge_args_yaml(parse_args())

    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)
    args.device = torch.device("cuda")
    
    main(args)