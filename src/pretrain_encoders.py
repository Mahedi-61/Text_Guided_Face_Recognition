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
from models.losses import clip_loss, sent_loss, words_loss, CMPLoss
from utils.prepare import prepare_dataloader, prepare_train_data_for_Bert, prepare_train_data

from models.models import (BERT_ENCODER, RNN_ENCODER, CNN_ENCODER,
                        BERTHeading, ResNet18_ArcFace_ENCODER, ResNet18_ArcFace_Heading)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PretrainEncoders')
    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/pretrain_celeba.yml',
                        help='optional config file')
    parser.add_argument('--train', type=bool, default=True, help='if train model')
    args = parser.parse_args()
    return args


def save_encoders(text_encoder, text_head, image_encoder, image_head, 
                            optimizer, optimizer_head, args):

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

    if  args.is_second_step == False:
        step = "first"
    else:
        step = "second"
    torch.save(checkpoint_image_en, '%s/arcface_image_encoder_%s_%s_%d.pth' % 
                                    (save_dir, args.en_type, step, args.current_epoch))

    if text_head is not None: 
        checkpoint_text_en = {
            'model': text_encoder.state_dict(),
            'head': text_head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_head': optimizer_head.state_dict(),
        }
    else:
        checkpoint_text_en = {
            'model': text_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_head': optimizer_head.state_dict(),
        }

    
    torch.save(checkpoint_text_en, '%s/arcface_text_encoder_%s_%s_%d.pth' % 
                                    (save_dir, args.en_type, step, args.current_epoch))



def train(dataloader,  text_encoder, text_head, image_encoder, image_head, optimizer,  
            lr_scheduler, optimizer_head, cmp_loss, args):

    text_encoder.train()
    if text_head is not None: text_head.train() 
    image_encoder.train()
    image_head.train()

    batch_size = args.batch_size 
    labels = prepare_labels(batch_size)
    epoch = args.current_epoch 
    total_length = len(dataloader)

    s_total_loss = 0
    w_total_loss = 0
    
    total_cl_loss = 0
    total_cmp_loss = 0
    total_damsm_loss = 0

    loop = tqdm(total = total_length)
    for step, data in enumerate(dataloader, 0):  
        """
        Text Encoder
        """     
        if args.using_BERT == True:
            imgs, words_emb, word_vector, sent_emb, keys, class_ids = \
                    prepare_train_data_for_Bert(data, text_encoder, text_head)
            cap_lens = None 

        if args.using_BERT == False:
            imgs, words_emb, sent_emb, keys, class_ids, cap_lens = \
                    prepare_train_data(data, text_encoder, text_head)

        """
        Image Encoder
        words_features: batch_size x nef x 17 x 17
        sent_code: batch_size x nef
        """
        words_features, sent_code = image_encoder(imgs)
        words_features, sent_code = image_head(words_features, sent_code)
        
        optimizer.zero_grad()
        optimizer_head.zero_grad() 
        total_loss = 0
        if args.is_DAMSM == True:
            w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                cap_lens, class_ids.numpy(), batch_size, args)
            
            
            s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids.numpy(), batch_size, args)

            damsm_loss = w_loss0 + w_loss1 + s_loss0 + s_loss1
            total_damsm_loss += damsm_loss
            total_loss += damsm_loss 
            w_total_loss += (w_loss0 + w_loss1).data / 2
            s_total_loss += (s_loss0 + s_loss1).data / 2

        #clip loss
        if args.is_CLIP == True:
            cl_loss = clip_loss(sent_emb, sent_code, args) 
            total_loss += args.lambda_cl * cl_loss
            total_cl_loss += args.lambda_cl * cl_loss  

        # cross-modal projection loss
        if args.is_CMP == True: 
            class_ids = class_ids.cuda() 
            cmp, cmpc, cmpm = cmp_loss(sent_emb, sent_code, class_ids)
            total_loss += cmp
            total_cmp_loss += cmp

        # update
        total_loss.backward()
        if args.is_second_step == False: 
            optimizer.step()
        optimizer_head.step()
        lr_scheduler.step()

        if (step % 5000 == 0):
            print("step:", step)
            print(lr_scheduler.get_lr())

        #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.        
        torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), args.clip_max_norm)

        # update loop information
        loop.update(1)
        loop.set_description(f'Training Epoch [{epoch}/{args.max_epoch}]')
        loop.set_postfix()
    loop.close()


    print(' | epoch {:3d} |' .format(args.current_epoch))
    if args.is_DAMSM == True:
        total_damsm_loss = total_damsm_loss.detach().cpu().numpy() / total_length
        s_total_loss = s_total_loss / total_length
        w_total_loss = w_total_loss / total_length
        print('s_loss {:5.4f} | w_loss {:5.4f} | DAMSM loss {:5.4f}'.format(s_total_loss, w_total_loss, total_damsm_loss))

    if args.is_CLIP == True:
        print("Total clip loss: {:5.4f} ".format(total_cl_loss.detach().cpu().numpy() / total_length))

    if args.is_CMP == True: 
        print("Total cmp loss: {:5.4f} ".format(total_cmp_loss.detach().cpu().numpy() / total_length))



def evaluate(dataloader, text_encoder, text_head, image_encoder, image_head, args):
    text_encoder.eval()
    if text_head is not None: text_head.eval() 
    image_encoder.eval()
    image_head.eval()

    s_total_loss = 0
    w_total_loss = 0
    total_cl_loss = 0

    batch_size = args.batch_size
    labels = prepare_labels(batch_size) 
    loop = tqdm(total = len(dataloader))

    with torch.no_grad():
        for step, data in enumerate(dataloader, 0):
            if args.using_BERT == True:
                imgs, words_emb, word_vector, sent_emb, keys, class_ids = \
                        prepare_train_data_for_Bert(data, text_encoder, text_head)
                cap_lens = None 

            if args.using_BERT == False:
                imgs, words_emb, sent_emb, keys, class_ids, cap_lens = \
                        prepare_train_data(data, text_encoder, text_head)

            """
            Image Encoder
            words_features: batch_size x nef x 17 x 17
            sent_code: batch_size x nef
            """
            words_features, sent_code = image_encoder(imgs)
            words_features, sent_code = image_head(words_features, sent_code)

            if args.is_DAMSM == True:
                w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids.numpy(), batch_size, args)

                s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids.numpy(), batch_size, args)
                w_total_loss += (w_loss0 + w_loss1).data
                s_total_loss += (s_loss0 + s_loss1).data

            #clip loss
            if args.is_CLIP == True:
                cl_loss = clip_loss(sent_emb, sent_code, args) 
                total_cl_loss += cl_loss  

            loop.update(1)

    loop.close()
    s_total_loss = s_total_loss / step
    w_total_loss = w_total_loss / step
    total_cl_loss = total_cl_loss / step
    return s_total_loss, w_total_loss, total_cl_loss



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
    image_encoder = ResNet18_ArcFace_ENCODER(args)
    image_encoder = nn.DataParallel(image_encoder, device_ids=args.gpu_id).cuda()
    image_head = ResNet18_ArcFace_Heading(args)
    image_head = nn.DataParallel(image_head, device_ids=args.gpu_id).cuda() 

    # building text encoder
    if args.using_BERT == True:
        text_encoder = BERT_ENCODER(args)
        text_encoder = nn.DataParallel(text_encoder, device_ids=args.gpu_id).cuda() 
        text_head = BERTHeading(args)
        text_head = nn.DataParallel(text_head, device_ids=args.gpu_id).cuda()

        params = [
            #{"params": image_encoder.parameters(), "lr": args.lr_image},
            {"params": text_encoder.parameters(), "lr": args.lr_text_bert},
        ]

        params_head = [
            {"params": itertools.chain(text_head.parameters(), image_head.parameters()),
             "lr": args.lr_head}
        ]

    elif args.using_BERT == False:
        text_encoder = RNN_ENCODER(args, nhidden=args.embedding_dim)
        text_encoder = text_encoder.cuda()
        text_head = None  

        params = [
            #{"params": image_encoder.parameters(), "lr": args.lr_image},
            {"params": text_encoder.parameters(), "lr": args.lr_text},
        ]

        params_head = [
            {"params": itertools.chain(text_head.parameters(), image_head.parameters()),
             "lr": args.lr_head}
        ]

    optimizer = torch.optim.AdamW(params,  betas=(0.9, 0.999), weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(params,  betas=(0.9, 0.999), weight_decay=args.weight_decay) 
    optimizer_head = torch.optim.Adam(params_head,  betas=(0.5, 0.999)) 

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5)
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #        optimizer, mode="min", patience=args.patience, factor=args.factor)

    lr_scheduler_head = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_head, mode="min", patience=args.patience, factor=args.factor)
        
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
        optimizer_head.load_state_dict(state_dict['optimizer_head'])

        print('Load ', args.resume_model_path)
        name = args.resume_model_path.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)

        image_encoder.load_state_dict(state_dict['model'])
        image_head.load_state_dict(state_dict["head"])
        print('Load ', name)

    return (text_encoder, text_head, image_encoder, image_head, optimizer, 
            lr_scheduler, optimizer_head, lr_scheduler_head, start_epoch)



def prepare_labels(batch_size):
    match_labels = Variable(torch.LongTensor(range(batch_size)))
    return match_labels.cuda()


def main(args):
    ######################### Get data loader ########################
    imsize = args.img_size
    image_transform =   transforms.Compose([
                        transforms.RandomCrop(imsize),
                        transforms.RandomHorizontalFlip()])

    print("Loading training and valid data ...")
    train_dl, train_ds, valid_dl, valid_ds = prepare_dataloader(args, "train", image_transform)
    
    args.len_train_dl = len(train_dl)
    if args.using_BERT == False:
        print("dataset words: %s, embeddings number: %s" % (train_ds.n_words, train_ds.embeddings_num))
        args.vocab_size = train_ds.n_words

    del train_ds, valid_ds
    
    # Build model
    if args.prev_weight == False:
        text_encoder, text_head, image_encoder, image_head, optimizer, \
                            lr_scheduler, optimizer_head, lr_scheduler_head, start_epoch = build_models(args)
    
    elif args.prev_weight == True:
        text_encoder, image_encoder, optimizerT, optimizerI, \
            lr_schedulerT, lr_schedulerI, start_epoch= build_models_with_prev_weight(args)

   
    if args.is_CMP == True: 
        # initialize losses
        cmp_loss = CMPLoss(is_CMPM=False, is_CMPC=True, num_classes=24000)
        cmp_loss.cuda()
    else:
        cmp_loss = 0

    for epoch in range(start_epoch, args.max_epoch + 1):
        args.current_epoch = epoch

        print('Reset scheduler')
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5)
        train(train_dl, text_encoder, text_head, image_encoder, image_head, optimizer, lr_scheduler,
                optimizer_head, cmp_loss, args)


        if ((args.do_test == True) and (epoch % args.test_interval == 0)):
            print("Let's evaluate the model")

            s_loss, w_loss, total_clip_loss = evaluate(valid_dl, text_encoder, text_head, image_encoder, image_head, args)
            total_loss = s_loss + w_loss + total_clip_loss 
            lr_scheduler_head.step(total_loss)
            print('| end epoch {:3d} | valid loss {:5.4f} {:5.4f} {:5.4f}'.format(epoch, s_loss, w_loss, total_clip_loss))

        if (epoch % args.save_interval == 0 or epoch == args.max_epoch):
            print("saving image and text encoder\n")
            save_encoders(text_encoder, text_head, image_encoder, image_head, 
                            optimizer, optimizer_head, args)

        
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