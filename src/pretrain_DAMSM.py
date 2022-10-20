from __future__ import print_function
import os
import sys
import os.path as osp
import random
import pprint
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from tqdm import tqdm 
import itertools

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)

from utils.utils import mkdir_p, merge_args_yaml
from models.losses import clip_loss, sent_loss, words_loss
from utils.prepare import prepare_dataloader, prepare_train_data_for_Bert, prepare_train_data

from models.models import (BERT_ENCODER, RNN_ENCODER, CNN_ENCODER, 
                        BERTHeading, ResNetFace_ENCODER, ResNetFace_Heading)



def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='DAMSM')
    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/DAMSM_celeba.yml',
                        help='optional config file')
    parser.add_argument('--train', type=bool, default=True, help='if train model')
    args = parser.parse_args()
    return args


def save_encoders(text_encoder, text_head, image_encoder, image_head, 
                          optimizer, lr_scheduler, args):

    if args.using_BERT == True: folder = "Bert"
    elif args.using_BERT == False: folder = "BiLSTM"

    save_dir = os.path.join(args.checkpoints_path, 
                            args.dataset_name, 
                            args.CONFIG_NAME, folder)
    mkdir_p(save_dir)

    checkpoint_image_en = {
        'model': image_encoder.state_dict(),
        "head": image_head.state_dict()
    }
    torch.save(checkpoint_image_en, '%s/arc_image_encoder_%s_%d.pth' % 
                                    (save_dir, args.en_type, args.current_epoch))

    checkpoint_text_en = {
        'model': text_encoder.state_dict(),
        'head': text_head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }
    torch.save(checkpoint_text_en, '%s/arc_text_encoder_%s_%d.pth' % 
                                    (save_dir, args.en_type, args.current_epoch))



def train(dataloader,  text_encoder, text_head, image_encoder, image_head, optimizer, args):
    batch_size = args.train_batch_size 
    labels = prepare_labels(batch_size)
    epoch = args.current_epoch 

    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    total_length = len(dataloader)
    total_cl_loss = 0

    loop = tqdm(total = total_length)
    for step, data in enumerate(dataloader, 0):        
        if args.using_BERT == True:
            imgs, words_emb, sent_emb, keys, class_ids = \
                    prepare_train_data_for_Bert(data, text_encoder, text_head)
            cap_lens = None 

        if args.using_BERT == False:
            imgs, words_emb, sent_emb, keys, class_ids, cap_lens = \
                    prepare_train_data(data, text_encoder, text_head)
        

        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        words_features, sent_code = image_encoder(imgs)
        words_features, sent_code = image_head(words_features, sent_code )

        # calculate loss 
        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids.numpy(), batch_size, args)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        damsm_loss = w_loss0 + w_loss1 

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids.numpy(), batch_size, args)
        damsm_loss += s_loss0 + s_loss1 
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data

        cl = clip_loss(sent_emb, sent_code, args)        
        damsm_loss += cl

        optimizer.zero_grad() 
        damsm_loss.backward()
        optimizer.step()

        #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.        
        #torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), args.clip_max_norm)

        total_cl_loss += cl 
        # update loop information
        loop.update(1)
        loop.set_description(f'Training Epoch [{epoch}/{args.max_epoch}]')
        loop.set_postfix()

    loop.close()

    s_cur_loss0 = s_total_loss0 / total_length
    s_cur_loss1 = s_total_loss1 / total_length
    w_cur_loss0 = w_total_loss0 / total_length
    w_cur_loss1 = w_total_loss1 / total_length

    print('| epoch {:3d} | s_loss {:5.2f} {:5.2f} | w_loss {:5.2f} {:5.2f}'.
            format(args.current_epoch, s_cur_loss0, s_cur_loss1, w_cur_loss0, w_cur_loss1))

    print("Total clip loss: ", total_cl_loss / total_length)



def evaluate(dataloader, text_encoder, text_head, image_encoder, image_head, args):
    text_encoder.eval()
    text_head.eval() 
    image_encoder.eval()
    image_head.eval()
    s_total_loss = 0
    w_total_loss = 0

    batch_size = args.test_batch_size
    labels = prepare_labels(batch_size) 

    with torch.no_grad():
        for step, data in enumerate(dataloader, 0):
            if args.using_BERT == True:
                imgs, words_emb, sent_emb, keys, class_ids = \
                        prepare_train_data_for_Bert(data, text_encoder, text_head)
                cap_lens = None 

            if args.using_BERT == False:
                imgs, words_emb, sent_emb, keys, class_ids, cap_lens = \
                        prepare_train_data(data, text_encoder, text_head)

            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            words_features, sent_code = image_encoder(imgs)
            words_features, sent_code = image_head(words_features, sent_code)

            w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                        cap_lens, class_ids.numpy(), batch_size, args)

            w_total_loss += (w_loss0 + w_loss1).data
            s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids.numpy(), batch_size, args)
            s_total_loss += (s_loss0 + s_loss1).data

            if step == 300:
                break

    s_cur_loss = s_total_loss / step
    w_cur_loss = w_total_loss / step
    return s_cur_loss, w_cur_loss



def build_models_with_prev_weight(args):
    # building text encoder
    text_encoder = RNN_ENCODER(args, nhidden=args.embedding_dim)
    optimizerT = torch.optim.AdamW(text_encoder.parameters(), 
                                lr = args.lr_lstm, 
                                weight_decay = args.weight_decay)

    lr_schedulerT = torch.optim.lr_scheduler.StepLR(optimizerT, 
                                                    args.lr_drop, 
                                                    gamma = args.lr_gamma)

    # building image encoder
    image_encoder = CNN_ENCODER(args)
    para = []
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)

    optimizerI = torch.optim.AdamW(para, lr = args.lr, 
                                        weight_decay = args.weight_decay)

    lr_schedulerI = torch.optim.lr_scheduler.StepLR(optimizerI, 
                                        args.lr_drop, 
                                        gamma=args.lr_gamma)

    # loading checkpoints
    print("loading checkpoint; epoch: ", args.resume_epoch)
    start_epoch = args.resume_epoch + 1
    text_encoder.load_state_dict(torch.load(args.resume_model_path, map_location="cpu"))
    text_encoder = text_encoder.cuda()
    print('Load ', args.resume_model_path)

    name = args.resume_model_path.replace('text_encoder', 'image_encoder')
    image_encoder.load_state_dict(torch.load(name, map_location="cpu"))
    image_encoder = nn.DataParallel(image_encoder, device_ids=args.gpu_id).cuda()
    print('Load ', name)

    return (text_encoder, image_encoder, optimizerT, optimizerI, 
            lr_schedulerT, lr_schedulerI, start_epoch)



def build_models(args):
    # building image encoder
    image_encoder = ResNetFace_ENCODER(args)
    image_encoder = nn.DataParallel(image_encoder, device_ids=args.gpu_id).cuda()
    image_head = ResNetFace_Heading(args)
    image_head = nn.DataParallel(image_head, device_ids=args.gpu_id).cuda() 

    # building text encoder
    if args.using_BERT == True:
        text_encoder = BERT_ENCODER(args)
        text_encoder = nn.DataParallel(text_encoder, device_ids=args.gpu_id).cuda()
        text_head = BERTHeading(args)
        text_head = nn.DataParallel(text_head, device_ids=args.gpu_id).cuda()

        params = [
            {"params": image_encoder.parameters(), "lr": args.lr_image},
            {"params": text_encoder.parameters(), "lr": args.lr_text_bert},
            {"params": itertools.chain(text_head.parameters(), image_head.parameters()),
             "lr": args.lr_head, "weight_decay": args.weight_decay}
        ]

    elif args.using_BERT == False:
        text_encoder = RNN_ENCODER(args, nhidden=args.embedding_dim)
        text_encoder = text_encoder.cuda()

        params = [
            {"params": image_encoder.parameters(), "lr": args.lr_image},
            {"params": text_encoder.parameters(), "lr": args.lr_text},
            {"params": image_head.parameters(), 
             "lr": args.lr_head, "weight_decay": args.weight_decay}
        ]

    optimizer = torch.optim.AdamW(params, weight_decay=0.0)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=args.patience, factor=args.factor)
        
    
    # load from checkpoint
    start_epoch = 1
    if args.resume_epoch!=1:
        print("loading checkpoint; epoch: ", args.resume_epoch)
        start_epoch = args.resume_epoch + 1

        state_dict = torch.load(args.resume_model_path)
        text_encoder.load_state_dict(state_dict['model'])

        if args.using_BERT == True:
            text_head.load_state_dict(state_dict["head"])
        else:
            text_head = None 

        optimizer.load_state_dict(state_dict['optimizer'])
        lr_scheduler.load_state_dict(state_dict['lr_scheduler'])

        print('Load ', args.resume_model_path)
        name = args.resume_model_path.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)

        image_encoder.load_state_dict(state_dict['model'])
        image_head.load_state_dict(state_dict["head"])
        print('Load ', name)

    return (text_encoder, text_head, image_encoder, image_head, optimizer, lr_scheduler, start_epoch)



def prepare_labels(batch_size):
    match_labels = Variable(torch.LongTensor(range(batch_size)))
    return match_labels.cuda()


def main(args):
    ######################### Get data loader ########################
    imsize = args.img_size
    image_transform =   transforms.Compose([
                        #transforms.Resize(int(imsize * 76 / 64)),
                        transforms.RandomCrop(imsize),
                        transforms.RandomHorizontalFlip()])

    print("Loading training and valid data ...")
    train_dl, train_ds, valid_dl, valid_ds = prepare_dataloader(args, "train", image_transform)

    args.len_train_dl = len(train_dl)
    if args.using_BERT == False:
        print("dataset words: %s, embeddings number: %s" % 
                            (train_ds.n_words, train_ds.embeddings_num))
        args.vocab_size = train_ds.n_words

    
    # Build model
    if args.prev_weight == False:
        text_encoder, text_head, image_encoder, image_head, optimizer, \
                            lr_scheduler, start_epoch = build_models(args)
    
    elif args.prev_weight == True:
        text_encoder, image_encoder, optimizerT, optimizerI, \
            lr_schedulerT, lr_schedulerI, start_epoch= build_models_with_prev_weight(args)


    for epoch in range(start_epoch, args.max_epoch + 1):
        args.current_epoch = epoch
        text_encoder.train()
        text_head.train() 
        image_encoder.train()
        image_head.train()

        train(train_dl, text_encoder, text_head, image_encoder, image_head, optimizer, args)
        
        if ((args.do_test == True) and (epoch % args.test_interval == 0)):
            print("Let's evaluate the model")

            s_loss, w_loss = evaluate(valid_dl, text_encoder, text_head, image_encoder, image_head, args)
            print('| end epoch {:3d} | valid loss {:5.2f} {:5.2f}'.format(epoch, s_loss, w_loss))

        if (epoch % args.save_interval == 0 or epoch == args.max_epoch):
            print("saving image and text encoder\n")
            save_encoders(text_encoder, text_head, image_encoder, image_head, 
                          optimizer, lr_scheduler, args)


        lr_scheduler.step(s_loss)



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