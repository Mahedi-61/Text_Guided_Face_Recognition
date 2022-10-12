from os import pread
import sys
import os.path as osp
import random
import argparse
import numpy as np
import pprint
import torch
from tqdm import tqdm 


ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils.utils import mkdir_p, merge_args_yaml
from utils.utils import save_models, load_model_opt 
from utils.prepare import prepare_dataloader, prepare_text_encoder, prepare_models
from utils.modules import test, get_features  
from utils.train_dataset import get_one_batch_data, prepare_train_data, get_one_batch_data_Bert
from models import metrics, focal_loss

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
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
        metric_fc = metrics.AddMarginProduct(1024, 
                                            args.num_classes, 
                                            s=30, m=0.35)
    elif args.metric == "arc_margin":
        metric_fc = metrics.ArcMarginProduct(1024, 
                                            args.num_classes, 
                                            s=30, m=0.5, 
                                            easy_margin=args.easy_margin)

    elif args.metric == "linear":
        metric_fc = metrics.MyLinear(1024, args.num_classes)
    
    metric_fc.to(args.device)
    metric_fc = torch.nn.DataParallel(metric_fc, device_ids=args.gpu_id)
    return metric_fc


def get_optimizer(args, netG, metric_fc):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': netG.parameters()}, 
                                     {'params': metric_fc.parameters()}],
                                     lr=args.lr_image_train, 
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': netG.parameters()}, 
                                      {'params': metric_fc.parameters()}],
                                      lr=args.lr_image_train, 
                                      weight_decay=args.weight_decay)
    return optimizer

    

def train(train_dl, model, netG, metric_fc, text_encoder, criterion, optimizer, args):
    device = args.device
    epoch = args.current_epoch
    max_epoch = args.max_epoch
    netG = netG.train()
    metric_fc = metric_fc.train()

    loop = tqdm(total=len(train_dl))
    for step, data in enumerate(train_dl, 0):
        imgs, sent_emb, words_embs, keys, label = prepare_train_data(data, text_encoder)
        imgs = imgs.to(device).requires_grad_()
        sent_emb = sent_emb.to(device).requires_grad_()
        #words_embs = words_embs.to(device).requires_grad_()
        label = label.to(device)
        
        img_features = get_features(model, imgs)
        output = netG(img_features, sent_emb)
        
        output = metric_fc(output, label)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update loop information
        loop.update(1)
        loop.set_description(f'Training Epoch [{epoch}/{max_epoch}]')
        loop.set_postfix()

    loop.close()
    del img_features


def main(args):
    # prepare dataloader, models, data
    args.model_save_file = osp.join(args.checkpoints_path, str(args.dataset_name))
    mkdir_p(args.model_save_file)

    train_dl, train_ds, valid_dl, valid_ds = prepare_dataloader(args, split="train", transform=None)
    #test_dl, test_ds = get_test_dataloader(args)

    get_one_batch_data_Bert(train_dl)

    """
    args.vocab_size = train_ds.n_words
    text_encoder = prepare_text_encoder(args)
    
    model, netG = prepare_models(args)
    metric_fc = get_margin(args)

    optimizer = get_optimizer(args, netG, metric_fc)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                    step_size=args.lr_step, 
                                    gamma=0.1)

    criterion = get_loss(args)
    
    # load from checkpoint
    strat_epoch = 1
    if args.resume_epoch!=1:
        print("loading checkpoint; epoch: ", args.resume_epoch)
        strat_epoch = args.resume_epoch+1
        netG, metric_fc, optimizer = load_model_opt(netG, metric_fc, optimizer, args.resume_model_path)
    

    #pprint.pprint(args)
    print("Start Training")
    for epoch in range(strat_epoch, args.max_epoch + 1):
        torch.cuda.empty_cache()
        args.current_epoch = epoch 
        train(train_dl, model, netG, metric_fc, text_encoder, criterion, optimizer, args)
        
        # save
        if epoch % args.save_interval==0:
            save_models(netG, metric_fc, optimizer, epoch, args)
            scheduler.step()
            print("learning rate: ", optimizer.param_groups[0]['lr'])
        
        if ((args.do_test == True) and (epoch % args.test_interval == 0)):
            print("Let's test the model")
            test(test_dl, model, netG, text_encoder, args)
    """

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