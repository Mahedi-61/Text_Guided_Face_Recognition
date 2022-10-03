import sys
import os.path as osp
import random
import argparse
import os 
import numpy as np
import pprint
import torch
from tqdm import tqdm 


ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils.utils import merge_args_yaml, mkdir_p, load_full_model_for_image_rec
from utils.prepare import get_train_dataloader, get_test_dataloader
from models import resnet, metrics, focal_loss
from image_test import test 

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='ImgRec')
    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/FE_celeba.yml',
                        help='optional config file')
    parser.add_argument('--train', type=bool, default=True, help='if train model')
    args = parser.parse_args()
    return args


def get_model(args):
    if args.backbone == 'resnet18':
        if args.dataset_name == "celeba":
            model = resnet.resnet_face18(use_se=args.use_se)

        elif args.dataset_name == "birds":
            model = resnet.resnet_face18(use_se=args.use_se)

    elif args.backbone == 'resnet34':
        model = resnet.resnet34()
    elif args.backbone == 'resnet50':
        model = resnet.resnet50()

    model.to(args.device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
    return model 


def get_margin(args):
    if args.metric == "add_margin":
        metric_fc = metrics.AddMarginProduct(512, 
                                            args.num_classes, 
                                            s=30, m=0.35)
    elif args.metric == "arc_margin":
        metric_fc = metrics.ArcMarginProduct(512, 
                                            args.num_classes, 
                                            s=30, m=0.5, 
                                            easy_margin=args.easy_margin)
    elif args.metric == "sphere":
        metric_fc = metrics.SphereProduct(512, 
                                          args.num_classes, 
                                          m=4)
    elif args.metric == "linear":
        metric_fc = metrics.MyLinear(512, args.num_classes)
    
    metric_fc.to(args.device)
    metric_fc = torch.nn.DataParallel(metric_fc, device_ids=args.gpu_id)
    return metric_fc


def get_optimizer(args, model, metric_fc):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, 
                                     {'params': metric_fc.parameters()}],
                                     lr=args.lr_image_train, 
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}, 
                                      {'params': metric_fc.parameters()}],
                                      lr=args.lr_image_train, 
                                      weight_decay=args.weight_decay)
    return optimizer


def get_loss(args):
    if args.loss == "focal_loss":
        criterion = focal_loss.FocalLoss(gamma=2)

    elif args.loss == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()

    return criterion


def save_full_model_for_image_rec(model, metric_fc, optimizer, epoch):
    model_file_name = args.backbone + '_' + args.dataset_name + '_' + str(epoch) + '.pth'
    model_dir = os.path.join(args.checkpoints_path, args.dataset_name, args.CONFIG_NAME)
    mkdir_p(model_dir)
    save_name = os.path.join(model_dir, model_file_name)

    state = {'full_model': {'model': model.state_dict(), 
                            'metric_fc': metric_fc.state_dict()}, 
             'optimizer': {'optimizer': optimizer.state_dict()}}

    torch.save(state, save_name)
    return save_name




def train(train_dl, model, metric_fc, criterion, optimizer, scheduler, args):
    model.train()
    metric_fc.train()
    print(f'\nTraining Epoch [{args.current_epoch}/{args.max_epoch}]')

    loop = tqdm(train_dl, leave=True)
    for data in loop:
        data_input, caps, cap_len, key, label = data
        data_input = data_input.to(args.device)
        label = label.to(args.device).long()
        
        feature = model(data_input)
        output = metric_fc(feature, label)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("loss {:0.4f}".format(loss.item()))
    scheduler.step()
    loop.close()



def main(args):
    train_dl, train_ds = get_train_dataloader(args)

    args.vocab_size = train_ds.n_words
    print('{} train iters per epoch:'.format(len(train_dl)))

    model = get_model(args)
    metric_fc = get_margin(args)
    optimizer = get_optimizer(args, model, metric_fc)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                    step_size=args.lr_step, 
                                    gamma=0.1)

    criterion = get_loss(args)

    
    # load from checkpoint
    start_epoch = 1
    if args.resume_epoch!=1:
        print("loading from checkpoint | epoch: ", args.resume_epoch)
        start_epoch = args.resume_epoch+1
        model, metric_fc, optimizer = load_full_model_for_image_rec(model, 
                                metric_fc, optimizer, args.resume_model_path)
    

    #pprint.pprint(args)
    print("Start Training .............. bismillah...............")
    for epoch in range(start_epoch, args.max_epoch + 1):
        torch.cuda.empty_cache()
        args.current_epoch = epoch

        train(train_dl, model, metric_fc, criterion, optimizer, scheduler, args)
        
        # save
        if epoch % args.save_interval==0 or epoch == args.max_epoch:
            save_full_model_for_image_rec(model, metric_fc, optimizer, epoch)
        
    
        if ((args.do_test == True) and (epoch % args.test_interval == 0)):
            test_dl, test_ds = get_test_dataloader(args)
            args.vocab_size = test_ds.n_words

            test(test_dl, model, args)


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

#python3 -m torch.distributed.launch --nproc_per_node=2 src/train.py