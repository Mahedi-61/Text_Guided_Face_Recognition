from __future__ import print_function

import torch
import torch.nn.functional as F
import argparse
import numpy as np
import random
import time
from models import resnet, metrics, focal_loss
from fr_test import *
from utils import visualizer, view_model
from utils.utils import merge_args_yaml, mkdir_p
import os
from fr_dataset import Dataset
from utils.perpare import prepare_dataloaders

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='archface')
    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/birds.yml',
                        help='optional config file')

    args = parser.parse_args()
    return args

def save_model(model, save_path, name, dataset, iter_cnt):
    model_file_name = name + '_' + dataset + '_' + str(iter_cnt) + '.pth'
    model_dir = os.path.join(save_path, dataset)
    mkdir_p(model_dir)
    save_name = os.path.join(model_dir, model_file_name)
    torch.save(model.state_dict(), save_name)
    return save_name

def main(args):
    if args.display:
        visualizer = visualizer.Visualizer()
    device = torch.device("cuda")
    
    train_dl, test_dl, train_ds, test_ds = prepare_dataloaders(args)
    """
    train_dataset = Dataset(args.train_root, args.train_list, 
                            phase='train')
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)

    
    identity_list = get_test_list(args.test_pair_list)
    img_paths = [os.path.join(args.test_root, each) for each in identity_list]
    """

    print('{} train iters per epoch:'.format(len(train_dl)))

    if args.loss == 'focal_loss':
        criterion = focal_loss.FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.backbone == 'resnet18':
        if args.CONFIG_NAME == "webface":
            model = resnet.resnet_face18(use_se=args.use_se)
        elif args.CONFIG_NAME == "birds":
            model = resnet.resnet_face18(use_se=args.use_se)

    elif args.backbone == 'resnet34':
        model = resnet.resnet34()
    elif args.backbone == 'resnet50':
        model = resnet.resnet50()

    if args.metric == 'add_margin':
        metric_fc = metrics.AddMarginProduct(512, args.num_classes, s=30, m=0.35)
    elif args.metric == 'arc_margin':
        metric_fc = metrics.ArcMarginProduct(512, args.num_classes, s=30, m=0.5, 
                                                easy_margin=args.easy_margin)
    elif args.metric == 'sphere':
        metric_fc = metrics.SphereProduct(512, args.num_classes, m=4)
    else:
        metric_fc = metrics.MyLinear(512, args.num_classes)


    # view_model(model, args.input_shape)
    model.to(device)
    model = torch.nn.DataParallel(model)
    metric_fc.to(device)
    metric_fc = torch.nn.DataParallel(metric_fc)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, 
                                     {'params': metric_fc.parameters()}],
                                     lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}, 
                                      {'params': metric_fc.parameters()}],
                                      lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                    step_size=args.lr_step, gamma=0.1)

    #start = time.time()
    print("Start Training .............. bismillah...............")
    for i in range(args.max_epoch):
        print("epoch: ", i+1)
        
        model.train()
        for ii, data in enumerate(train_dl):
            data_input, caps, cap_len, key, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if ii % 500 == 0: print(ii)

        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        label = label.data.cpu().numpy()

        acc = np.mean((output == label).astype(int))
        #speed = len(trainloader) / (time.time() - start)
        time_str = time.asctime(time.localtime(time.time()))
        
        print('{} train epoch {} ; loss {} ; acc {}'.format(time_str, i, loss.item(), acc))

        if args.display:
            visualizer.display_current_results(iters, loss.item(), name='train_loss')
            visualizer.display_current_results(iters, acc, name='train_acc')

        #start = time.time()
        scheduler.step()
        
        # saving models
        if i % args.save_interval == 0 or i == args.max_epoch:
            save_model(model, args.checkpoints_path, args.backbone, args.CONFIG_NAME, i)
        """
        # testing model
        if i % args.test_interval == 0 or i == args.max_epoch:
            model.eval()
            acc = test(model, img_paths, identity_list, args.test_pair_list, args.batch_size)

        if args.display:
            visualizer.display_current_results(iters, acc, name='test_acc') 
        """


if __name__ == '__main__':
    args = merge_args_yaml(parse_args())

    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    main(args)