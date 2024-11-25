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
from models.losses import sent_loss, words_loss, CMPLoss, ClipLoss, global_loss 
from utils.prepare import (prepare_dataloader, prepare_train_data_for_Bert, 
                           prepare_adaface, prepare_arcface)

from utils.modules import calculate_identification_acc, calculate_scores  
from models.models import (TextEncoder, TextHeading, ImageHeading)
from models import metrics, losses
from datetime import datetime 
from utils.prepare import prepare_test_data_Bert
today = datetime.now() 


def parse_args():
    # Training settings
    cfg_file = "train_bert.yml"
    print("loading %s" % cfg_file)
    parser = argparse.ArgumentParser(description='Train BERT Encoder')
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
        self.valid_dl, valid_ds = prepare_dataloader(self.args, split="valid", transform=None)
        print("Loading training and valid data ...")
        
        self.args.len_train_dl = len(self.train_dl)
        del train_ds, valid_ds 

        # Build model
        self.build_models()


    def save_encoders(self):
        save_dir = os.path.join(self.args.checkpoints_path, 
                                self.args.dataset_name, 
                                self.args.CONFIG_NAME, 
                                "BERT_" + self.args.model_type,
                                self.args.bert_type, today.strftime("%m-%d-%y-%H:%M"))
        mkdir_p(save_dir)

        checkpoint_image_en = {
            "image_head": self.image_head.state_dict()
        }

        torch.save(checkpoint_image_en, '%s/%s_image_encoder_%d.pth' % 
                    (save_dir, self.args.model_type, self.args.current_epoch))

        checkpoint_text_en = {
            'model': self.text_encoder.state_dict(),
            'head': self.text_head.state_dict()
        }

        torch.save(checkpoint_text_en, '%s/%s_text_encoder_%d.pth' % 
                    (save_dir, self.args.bert_type, self.args.current_epoch))



    def resume_checkpoint(self):
        print("loading checkpoint; epoch: ", self.args.resume_epoch)

        state_dict = torch.load(self.args.resume_model_path)
        self.text_encoder.load_state_dict(state_dict['model'])
        self.text_head.load_state_dict(state_dict['head'])

        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.optimizer_head.load_state_dict(state_dict['optimizer_head'])

        print('Load ', self.args.resume_model_path)
        name = self.args.resume_model_path.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)

        self.image_head.load_state_dict(state_dict["image_head"])
        print('Load ', name)


    def print_losses(self, s_total_loss, w_total_loss, total_cl_loss, total_cmp_loss, 
                     total_idn_loss, total_length):
        
        print(' | epoch {:3d} |' .format(self.args.current_epoch))
        if self.args.is_DAMSM == True:
            total_damsm_loss = (s_total_loss + w_total_loss) / total_length
            s_total_loss = s_total_loss / total_length
            w_total_loss = w_total_loss / total_length
            print('s_loss {:5.5f} | w_loss {:5.5f} | DAMSM loss {:5.5}'.format(s_total_loss, w_total_loss, total_damsm_loss)) 

        if self.args.is_CLIP == True:
            print("Total clip loss: {:5.6f} ".format(total_cl_loss.detach().cpu().numpy() / total_length))

        if self.args.is_CMP == True: 
            print("Total cmp loss: {:5.6f} ".format(total_cmp_loss / total_length))

        if self.args.is_ident_loss == True: 
            print("Total identity loss: {:5.6f} ".format(total_idn_loss / total_length))



    def build_image_encoders(self):
        if self.args.model_type == "arcface":
            self.image_encoder = prepare_arcface(self.args)
            self.image_head = ImageHeading(self.args)


        elif self.args.model_type == "adaface":
            self.image_encoder = prepare_adaface(self.args)
            self.image_head = ImageHeading(self.args)
            """"
            self.image_cls = metrics.AdaFace(self.args.aux_feat_dim_per_granularity,
                                            self.args.num_classes,
                                            m=0.4,
                                            h=0.333, 
                                            s=50)
            """

        self.image_cls = metrics.ArcMarginProduct(self.args.aux_feat_dim_per_granularity, 
                                                self.args.num_classes, 
                                                s=30, 
                                                m=0.5, 
                                                easy_margin=False)
            
        self.image_head = nn.DataParallel(self.image_head, 
                                device_ids=self.args.gpu_id).to(self.args.device)
            
        self.image_cls = torch.nn.DataParallel(self.image_cls, 
                                device_ids=self.args.gpu_id).to(self.args.device)
    

    def build_text_encoders(self):
        self.text_encoder = TextEncoder(self.args)
        self.text_encoder = nn.DataParallel(self.text_encoder, 
                                            device_ids=self.args.gpu_id).cuda() 
        self.text_head = TextHeading(self.args)
        self.text_head = nn.DataParallel(self.text_head, 
                                         device_ids=self.args.gpu_id).cuda()


        self.text_cls = metrics.ArcMarginProduct(self.args.aux_feat_dim_per_granularity, 
                                                self.args.num_classes, 
                                                s=35, 
                                                m=0.5, 
                                                easy_margin=False)
    
        self.text_cls = torch.nn.DataParallel(self.text_cls, 
                                              device_ids=self.args.gpu_id).to(self.args.device)


    def build_models(self):
        # building image encoder
        self.build_image_encoders() 
        
        # building text encoder
        self.build_text_encoders()
        
        # parameters
        params_head = [
            {"params": itertools.chain(self.text_head.parameters(), 
                                       self.image_head.parameters()),
            "lr": self.args.lr_head}
        ]

        params_cls = [
            {"params": itertools.chain(self.image_cls.parameters(), 
                                       self.text_cls.parameters())}
        ]

        # initialize losses
        self.ident_loss = losses.FocalLoss(gamma=2)
       
        if self.args.is_CMP == True: 
            self.cmp_loss = CMPLoss(is_CMPM = False, 
                               is_CMPC = True, 
                               num_classes = self.args.num_classes, 
                               feature_dim = self.args.aux_feat_dim_per_granularity)
            self.cmp_loss.cuda()

            params_head = [
            {"params": itertools.chain(self.text_head.parameters(), 
                                       self.image_head.parameters(),
                                       self.cmp_loss.parameters()),
            "lr": self.args.lr_head}
        ]
            

        if self.args.is_CLIP == True: 
            self.clip_loss = ClipLoss()

        self.optimizer_head = torch.optim.Adam(params_head,  betas=(0.5, 0.999)) 

        self.optimizer = torch.optim.Adam(self.text_encoder.parameters(),  
                                           betas=(0.9, 0.999),
                                           lr=self.args.min_lr_bert,
                                           weight_decay=self.args.weight_decay)
        
        self.optimizer_cls = torch.optim.SGD(params_cls, 
                                             lr=0.1, 
                                             momentum=0.9, 
                                             weight_decay=5e-5)


        self.lr_scheduler_head = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_head, 
                                                                        gamma=0.98)
           
        #resume checkpoint
        self.start_epoch = 1



    def train(self):
        self.text_encoder.train()
        self.text_head.train() 
        self.text_cls.train()
        self.image_head.train()
        self.image_cls.train()

        batch_size = self.args.batch_size 
        labels = self.prepare_labels(self.args.batch_size)
        epoch = self.args.current_epoch 
        total_length = len(self.train_dl) * self.args.batch_size

        s_total_loss = 0
        w_total_loss = 0
        
        total_cl_loss = 0
        total_cmp_loss = 0
        total_damsm_loss = 0

        loop = tqdm(total = len(self.train_dl))

        for  data in  self.train_dl:   
            imgs, caps, masks, keys, class_ids = data 
            words_emb, sent_emb = \
                    prepare_train_data_for_Bert((caps, masks), self.text_encoder, self.text_head)
            cap_lens = None 

            if self.args.model_type == "adaface":
                img_features, words_features, norm = self.image_encoder(imgs)
            else:
                img_features, words_features = self.image_encoder(imgs)

            img_features, words_features = self.image_head(img_features, words_features)

            self.optimizer.zero_grad()
            self.optimizer_head.zero_grad()
            self.optimizer_cls.zero_grad()  
            total_loss = 0

            if self.args.is_DAMSM == True:
                w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels, 
                                    cap_lens, class_ids.numpy(), batch_size, self.args)
                
                s_loss0, s_loss1 = sent_loss(img_features, sent_emb, labels, 
                                    class_ids.numpy(), batch_size, self.args)

                damsm_loss = w_loss0 + w_loss1 + s_loss0 + s_loss1 
                total_damsm_loss += damsm_loss.item()
                total_loss += damsm_loss 
                w_total_loss += ((w_loss0 + w_loss1).data).item()
                s_total_loss += ((s_loss0 + s_loss1).data).item()


            if self.args.is_WRA == True:
                pass 

            if self.args.is_ident_loss == True: 
                class_ids = class_ids.cuda() 
                # for text branch

                output = self.text_cls(sent_emb, class_ids)
                tid_loss = self.ident_loss(output, class_ids)

                # for image branch
                if self.args.model_type == "arcface":
                    output = self.image_cls(img_features, class_ids)
                elif self.args.model_type == "adaface":
                    output = self.image_cls(img_features, class_ids) #norm,

                iid_loss = self.ident_loss(output, class_ids)

                total_loss += self.args.lambda_id * tid_loss
                total_loss += self.args.lambda_id * iid_loss  
                total_idn_loss = self.args.lambda_id* iid_loss.item()

            # clip loss
            if self.args.is_CLIP == True:
                cl_loss = global_loss(img_features, sent_emb) 
                total_loss += self.args.lambda_clip * cl_loss
                total_cl_loss += self.args.lambda_clip * cl_loss  


            # cross-modal projection loss
            if self.args.is_CMP == True: 
                class_ids = class_ids.cuda() 
                cmp, _, __ = self.cmp_loss(sent_emb, img_features, class_ids)
                total_loss += cmp
                total_cmp_loss += cmp.item()

            # update
            total_loss.backward()
            self.optimizer.step()
            self.optimizer_head.step()
            self.optimizer_cls.step()
            #self.lr_scheduler.step()

            #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.        
            torch.nn.utils.clip_grad_norm_(itertools.chain(self.text_encoder.parameters()), 
                                                           self.args.clip_max_norm)

            # update loop information
            loop.update(1)
            loop.set_description(f'Training Epoch [{epoch}/{self.args.max_epoch}]')
            loop.set_postfix()

        loop.close()
        self.print_losses(s_total_loss, w_total_loss, total_cl_loss, 
                          total_cmp_loss, total_idn_loss, total_length)


    def prepare_labels(self, batch_size):
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        return match_labels.cuda()


    def test(self):
        device = self.args.device
        self.image_encoder.eval() 
        self.text_encoder.eval()
        preds = []
        g_truth = []

        with torch.no_grad():
            loop = tqdm(total=len(self.valid_dl))
            for data in self.valid_dl:
                img1, img2, words_emb1, words_emb2, sent_emb1, sent_emb2, \
                    pair_label = prepare_test_data_Bert(data, self.text_encoder, self.text_head)
                
                # upload to cuda
                img1 = img1.to(device)
                img2 = img2.to(device)
                pair_label = pair_label.to(device)

                if self.args.model_type == "arcface":
                    global_feat1,  local_feat1 = self.image_encoder(img1)
                    global_feat2,  local_feat2 = self.image_encoder(img2)
                    
                elif self.args.model_type == "adaface":
                    global_feat1,  local_feat1, norm = self.image_encoder(img1)
                    global_feat2,  local_feat2, norm = self.image_encoder(img2)

                sent_emb1 = sent_emb1.to(device)
                sent_emb2 = sent_emb2.to(device)

                proj_img_feat1, _ = self.image_head(global_feat1, local_feat1)
                proj_img_feat2, _ = self.image_head(global_feat2, local_feat2)

                out1 =  torch.cat((proj_img_feat1, sent_emb1), dim=1) 
                out2 =  torch.cat((proj_img_feat2, sent_emb2), dim=1) 
                del local_feat1, local_feat2, words_emb1, words_emb2

                cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
                pred = cosine_sim(out1, out2)
                preds += pred.data.cpu().tolist()
                g_truth += pair_label.data.cpu().tolist()

                # update loop information
                loop.update(1)
                loop.set_postfix()

        loop.close()
        calculate_scores(preds, g_truth, args)
        #calculate_identification_acc(preds, args)


    def main(self):
        LR_change_seq = [3, 8]
        lr = 0.1

        for epoch in range(self.start_epoch, self.args.max_epoch + 1):
            self.args.current_epoch = epoch

            self.train()
            self.lr_scheduler_head.step()
            if epoch in LR_change_seq: 
                for g in self.optimizer_cls.param_groups:
                    lr = lr * 0.1
                    g['lr'] = lr 
                    print("Learning Rate change to: {:0.5f}".format(lr))

            if (epoch % self.args.save_interval == 0 or epoch == self.args.max_epoch):
                print("saving image and text encoder\n")
                self.save_encoders()

            if (epoch > 12):
                if (epoch % self.args.test_interval == 0 and epoch !=0):
                    print("start validating")
                    self.args.is_roc = False   
                    self.test()



if __name__ == "__main__":
    args = merge_args_yaml(parse_args())

    # set seed
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)
    args.device = torch.device("cuda")
    Train(args).main()