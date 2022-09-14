from __future__ import print_function

import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision
import torch

import numpy as np
import random
import time
from config import Config
from models import resnet, metrics, focal_loss
from test import *
from utils import visualizer, view_model
import os
from dataset import Dataset


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


if __name__ == '__main__':

    opt = Config()
    if opt.display:
        visualizer = visualizer.Visualizer()
    device = torch.device("cuda")

    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)

    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    
    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = focal_loss.FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet.resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet.resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet.resnet50()

    if opt.metric == 'add_margin':
        metric_fc = metrics.AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = metrics.ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = metrics.SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    #print(model)
    model.to(device)
    model = torch.nn.DataParallel(model)
    metric_fc.to(device)
    metric_fc = torch.nn.DataParallel(metric_fc)

    
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()
    print("Start Training .............. bismillah...............")
    for i in range(opt.max_epoch):
        print("epoch: ", i+1)
        
        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
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
        # print(output)
        # print(label)
        acc = np.mean((output == label).astype(int))
        speed = len(trainloader) / (time.time() - start)
        time_str = time.asctime(time.localtime(time.time()))
        
        print('{} train epoch {} ; loss {} ; acc {}'.format(time_str, i, loss.item(), acc))
        if opt.display:
            visualizer.display_current_results(iters, loss.item(), name='train_loss')
            visualizer.display_current_results(iters, acc, name='train_acc')

        start = time.time()
        scheduler.step()
        
        # saving models
        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i)

        # testing model
        if i % opt.test_interval == 0 or i == opt.max_epoch:
            model.eval()
            acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)

        if opt.display:
            visualizer.display_current_results(iters, acc, name='test_acc') 