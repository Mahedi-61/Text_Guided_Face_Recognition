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
from tqdm import tqdm 
import itertools

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils.utils import mkdir_p, merge_args_yaml
from models.losses import clip_loss, sent_loss, words_loss, CMPLoss
from utils.prepare import prepare_dataloader, prepare_train_data, prepare_adaface, prepare_arcface

from models.models import (RNN_ENCODER, ArcFace_Heading)
from datetime import date
today = date.today() 


def parse_args():
    # Training settings
    cfg_file = "train_lstm.yml"
    print("loading %s" % cfg_file)
    parser = argparse.ArgumentParser(description='Train LSTM Encoder')
    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/%s' %cfg_file,
                        help='optional config file')
    args = parser.parse_args()
    return args


class Train:
    def __init__(self, args):
        self.args = args

        # prepare dataloader
        self.train_dl, train_ds = prepare_dataloader(self.args, split="train", transform=None)
        print("Loading training and valid data ...")
        
        self.args.len_train_dl = len(self.train_dl)
        print("dataset words: %s, embeddings number: %s" % (train_ds.n_words, train_ds.embeddings_num))
        self.args.vocab_size = train_ds.n_words
        del train_ds

        # Build model
        self.build_models()


    def save_encoders(self):
        save_dir = os.path.join(self.args.checkpoints_path, 
                                self.args.dataset_name, 
                                self.args.CONFIG_NAME, 
                                "BiLSTM_" + self.args.model_type, 
                                today.strftime("%m-%d-%y"))
        mkdir_p(save_dir)

        checkpoint_image_en = {
            "head": self.image_head.state_dict()
        }

        torch.save(checkpoint_image_en, '%s/%s_image_encoder_%d.pth' % 
                    (save_dir, self.args.model_type, self.args.current_epoch))


        checkpoint_text_en = {
            'model': self.text_encoder.state_dict(),
        }

        torch.save(checkpoint_text_en, '%s/%s_text_encoder_%d.pth' % 
                    (save_dir, self.args.en_type, self.args.current_epoch))


    def resume_checkpoint(self):
        print("loading checkpoint; epoch: ", self.resume_epoch)

        state_dict = torch.load(self.args.resume_model_path)
        self.text_encoder.load_state_dict(state_dict['model'])

        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.optimizer_head.load_state_dict(state_dict['optimizer_head'])

        print('Load ', self.args.resume_model_path)
        name = self.args.resume_model_path.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)

        self.image_head.load_state_dict(state_dict["head"])
        print('Load ', name)


    def print_losses(self, s_total_loss, w_total_loss, total_cl_loss, total_cmp_loss, total_length):
        print(' | epoch {:3d} |' .format(self.args.current_epoch))
        if self.args.is_DAMSM == True:
            total_damsm_loss = (s_total_loss + w_total_loss) / total_length
            s_total_loss = s_total_loss / total_length
            w_total_loss = w_total_loss / total_length
            print('s_loss {:5.5f} | w_loss {:5.5f} | DAMSM loss {:5.5}'.format(s_total_loss, w_total_loss, total_damsm_loss)) 

        if self.args.is_CLIP == True:
            print("Total clip loss: {:5.5f} ".format(total_cl_loss.detach().cpu().numpy() / total_length))

        if self.args.is_CMP == True: 
            print("Total cmp loss: {:5.5f} ".format(total_cmp_loss / total_length))


    def build_models(self):
        # building image encoder
        if self.args.model_type == "arcface":
            self.image_encoder = prepare_arcface(self.args)
            self.image_head = ArcFace_Heading(self.args)
            self.image_head = nn.DataParallel(self.image_head, device_ids=self.args.gpu_id).cuda() 

        elif self.args.model_type == "adaface":
            self.image_encoder = prepare_adaface(self.args)
            self.image_head = ArcFace_Heading(self.args)
            self.image_head = nn.DataParallel(self.image_head, device_ids=self.args.gpu_id).cuda() 


        # building text encoder
        self.text_encoder = RNN_ENCODER(self.args, nhidden=self.args.embedding_dim).cuda()


        params = [
            {"params": self.text_encoder.parameters(), "lr": self.args.init_lr_lstm},
        ]

        params_head = [
            {"params": itertools.chain(self.image_head.parameters()),
            "lr": self.args.lr_head}
        ]

        self.optimizer = torch.optim.AdamW(params,  
                                    betas=(0.9, 0.999), 
                                    weight_decay=self.args.weight_decay)
        
        #optimizer = torch.optim.Adam(params,  betas=(0.9, 0.999), weight_decay=args.weight_decay) 
        self.optimizer_head = torch.optim.Adam(params_head,  betas=(0.5, 0.999)) 

        self.lr_scheduler_head = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                                self.optimizer_head, mode="min", 
                                                                patience=self.args.patience, 
                                                                factor=self.args.factor)


        #resume checkpoint
        self.start_epoch = 1
        #self.start_epoch = self.args.resume_epoch + self.start_epoch
        #resume_checkpoint(self.args.resume_epoch, self.text_encoder, self.image_encoder, \
        # self.image_head, self.optimizer, self.optimizer_head)


    def prepare_labels(self, batch_size):
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        return match_labels.cuda()


    def train(self, cmp_loss):

        self.text_encoder.train()
        self.image_head.train()

        labels = self.prepare_labels(self.args.batch_size)
        epoch = self.args.current_epoch 
        total_length = len(self.train_dl) * self.args.batch_size

        s_total_loss = 0
        w_total_loss = 0
        
        total_cl_loss = 0
        total_cmp_loss = 0
        total_damsm_loss = 0

        loop = tqdm(total = len(self.train_dl))
        for data in self.train_dl:  
        
            """Text Encoder"""     
            imgs, words_emb, sent_emb, keys, class_ids, cap_lens = \
                        prepare_train_data(data, self.text_encoder)

            """
            Image Encoder
            words_features: batch_size x nef x 16 x 16 (arcface) [14x14 for adaface]
            sent_code: batch_size x nef
            """

            if self.args.model_type == "adaface":
                sent_code, words_features, norm = self.image_encoder(imgs)
            else:
                sent_code, words_features = self.image_encoder(imgs)

            sent_code, words_features = self.image_head(sent_code, words_features)

            self.optimizer.zero_grad()
            self.optimizer_head.zero_grad() 
            total_loss = 0

            if self.args.is_DAMSM == True:
                w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels, 
                                        cap_lens, class_ids.numpy(), self.args.batch_size, self.args)
                
                s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids.numpy(), self.args.batch_size, self.args)

                damsm_loss = s_loss0 + s_loss1 + w_loss0 + w_loss1 
                total_damsm_loss += damsm_loss.item()
                total_loss += damsm_loss 
                w_total_loss += ((w_loss0 + w_loss1).data).item()
                s_total_loss += ((s_loss0 + s_loss1).data).item()

            # clip loss
            if self.args.is_CLIP == True:
                cl_loss = clip_loss(sent_emb, sent_code, self.args) 
                total_loss += self.args.lambda_cl * cl_loss
                total_cl_loss += self.args.lambda_cl * cl_loss  

            # cross-modal projection loss
            if self.args.is_CMP == True: 
                class_ids = class_ids.cuda() 
                cmp, _, __ = cmp_loss(sent_emb, sent_code, class_ids)
                total_loss += cmp
                total_cmp_loss += cmp.item()

            # update
            total_loss.backward()
            self.optimizer.step()
            self.optimizer_head.step()
            self.lr_scheduler.step()

            #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.        
            torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(), self.args.clip_max_norm)

            # update loop information
            loop.update(1)
            loop.set_description(f'Training Epoch [{epoch}/{self.args.max_epoch}]')
            loop.set_postfix()

        loop.close()
        self.print_losses(s_total_loss, w_total_loss, total_cl_loss, total_cmp_loss, total_length)
        return total_damsm_loss + total_cl_loss + total_cmp_loss


    def main(self):

        if self.args.is_CMP == True: 
            # initialize losses
            cmp_loss = CMPLoss(is_CMPM = True, 
                               is_CMPC = True, 
                               num_classes = self.args.num_classes, 
                               feature_dim = self.args.aux_feat_dim_per_granularity)
            cmp_loss.cuda()
        else:
            cmp_loss = 0

        for epoch in range(self.start_epoch, self.args.max_epoch + 1):
            self.args.current_epoch = epoch

            print('Reset scheduler')
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                    T_max = self.args.r_step, 
                                                                    eta_min = self.args.min_lr_lstm)
            
            total_loss = self.train(cmp_loss)
            self.lr_scheduler_head.step(total_loss)

            if (epoch % self.args.save_interval == 0 or epoch == self.args.max_epoch):
                print("saving image and text encoder\n")
                self.save_encoders()


if __name__ == "__main__":
    args = merge_args_yaml(parse_args())

    # set seed
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)
    args.device = torch.device("cuda")
    Train(args).main()