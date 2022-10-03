from __future__ import print_function

import os
import sys
import os.path as osp
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from tqdm import tqdm 

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)

from utils.utils import mkdir_p, merge_args_yaml
from models.losses import sent_loss, words_loss

from utils.prepare import get_train_dataloader, get_test_dataloader
from utils.train_dataset import prepare_train_data_for_DAMSM

from models.DAMSM import RNN_ENCODER, CNN_ENCODER

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/DAMSM_celeba.yml',
                        help='optional config file')
    parser.add_argument('--train', type=bool, default=True, help='if train model')
    args = parser.parse_args()
    return args


def train(dataloader, cnn_model, rnn_model, labels, optimizer, epoch, args):
    cnn_model.train()
    rnn_model.train()
    batch_size = args.train_batch_size 

    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    total_length = len(dataloader)
    start_time = time.time()

    loop = tqdm(total = total_length)
    for step, data in enumerate(dataloader, 0):
        rnn_model.zero_grad()
        cnn_model.zero_grad()
        
        imgs, sent_emb, words_emb, keys, class_ids, cap_lens = \
                    prepare_train_data_for_DAMSM(data, rnn_model)

        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        words_features, sent_code = cnn_model(imgs)
        # --> batch_size x nef x 17*17
        nef, att_sze = words_features.size(1), words_features.size(2)
        # words_features = words_features.view(batch_size, nef, -1)

        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids.numpy(), batch_size, args)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids.numpy(), batch_size, args)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data
        loss.backward()

        
        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), args.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

        # update loop information
        loop.update(1)
        loop.set_description(f'Training Epoch [{epoch}/{args.TRAIN.MAX_EPOCH}]')
        loop.set_postfix()

    loop.close()

    s_cur_loss0 = s_total_loss0 / total_length
    s_cur_loss1 = s_total_loss1 / total_length

    w_cur_loss0 = w_total_loss0 / total_length
    w_cur_loss1 = w_total_loss1 / total_length

    print('| epoch {:3d} | s_loss {:5.2f} {:5.2f} | w_loss {:5.2f} {:5.2f}'.
            format(args.current_epoch, s_cur_loss0, s_cur_loss1, w_cur_loss0, w_cur_loss1))




def evaluate(dataloader, cnn_model, rnn_model, batch_size):
    cnn_model.eval()
    rnn_model.eval()
    s_total_loss = 0
    w_total_loss = 0

    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, cap_lens, class_ids, keys, \
                wrong_caps, wrong_caps_len, wrong_cls_id = prepare_data(data)

        words_features, sent_code = cnn_model(real_imgs[-1])

        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).data

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).data

        if step == 9:
            break

    s_cur_loss = s_total_loss / step
    w_cur_loss = w_total_loss / step

    return s_cur_loss, w_cur_loss


def build_models(args):
    # build model
    text_encoder = RNN_ENCODER(args, nhidden=args.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER()
    labels = Variable(torch.LongTensor(range(args.train_batch_size)))
    start_epoch = 0

    if args.TRAIN.NET_E != '':
        state_dict = torch.load(args.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load ', args.TRAIN.NET_E)
        
        name = args.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = args.TRAIN.NET_E.rfind('_') + 8
        iend = args.TRAIN.NET_E.rfind('.')
        start_epoch = args.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)

    if args.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = nn.DataParallel(image_encoder, device_ids=args.gpu_id).cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels, start_epoch



def main(args):
    save_dir = os.path.join(args.checkpoints_path, args.dataset_name, args.CONFIG_NAME)
    mkdir_p(save_dir)


    # Get data loader ##################################################
    imsize = args.img_size
    image_transform =   transforms.Compose([
                        transforms.Resize(int(imsize * 76 / 64)),
                        transforms.RandomCrop(imsize),
                        transforms.RandomHorizontalFlip()])

    print("Loading training data ...")
    train_dl, train_ds = get_train_dataloader(args, image_transform)
    print("dataset words: %s, embeddings number: %s" % (train_ds.n_words, train_ds.embeddings_num))


    ### test data ###
    print("loading test data ...")
    test_dl, test_ds = get_test_dataloader(args, image_transform)
    

    # Build model
    args.vocab_size = train_ds.n_words
    text_encoder, image_encoder, labels, start_epoch = build_models(args)
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)
    

    optimizer = optim.Adam(para, lr=args.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, 
                                    lr_lambda=lambda epoch: 0.98)

    for epoch in range(start_epoch, args.TRAIN.MAX_EPOCH):
        args.current_epoch = epoch 
        
        train(train_dl, image_encoder, text_encoder, labels, optimizer, epoch, args)
        print('-' * 89)

        """
        if len(dataloader_val) > 0:
            s_loss, w_loss = evaluate(dataloader_val, image_encoder,
                                    text_encoder, batch_size)
            print('| end epoch {:3d} | valid loss '
                '{:5.2f} {:5.2f} | lr {:.5f}|'
                .format(epoch, s_loss, w_loss, lr))
        """

        print('-' * 89)
        if (epoch < 115):
            scheduler.step()

        print("saving image and text encoder")
        if (epoch % args.save_interval == 0 or epoch == args.TRAIN.MAX_EPOCH):
            torch.save(image_encoder.state_dict(), '%s/image_encoder%d.pth' % (save_dir, epoch))
            torch.save(text_encoder.state_dict(),  '%s/text_encoder%d.pth' % (save_dir, epoch))


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