from torch.nn import Module, Parameter
import sys, math, os 
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
from utils.prepare import (prepare_dataloader, prepare_text_encoder, prepare_adaface, 
                           prepare_arcface, prepare_fusion_net, prepare_train_data)
from utils.modules import test, get_features_adaface, get_features
from models import metrics, losses
from models.models import AdaFace
from datetime import date 
today = date.today()


def parse_args():
    # Training settings
    cfg_file = "fusion_lstm.yml"
    print("Loading: ", cfg_file)
    parser = argparse.ArgumentParser(description='Fusion')
    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/%s' % cfg_file,
                        help='optional config file')
    args = parser.parse_args()
    return args


class Fusion:
    def __init__(self, args):

        self.args = args 
        # prepare dataloader
        self.train_dl, train_ds = prepare_dataloader(self.args, split="train", transform=None)
        self.valid_dl, valid_ds = prepare_dataloader(self.args, split="valid", transform=None)

        # preapare model
        self.args.vocab_size = train_ds.n_words
        del train_ds, valid_ds 
        self.text_encoder, self.text_head = prepare_text_encoder(self.args, test=False)
        
        if self.args.model_type == "arcface":
            self.model = prepare_arcface(self.args) #cuda + parallel + grd. false + eval
            
        elif self.args.model_type == "adaface":
            self.model = prepare_adaface(self.args)

        self.fusion_net = prepare_fusion_net(self.args) #cuda + parallel
        self.metric_fc = self.get_margin()

        # prepare loss & optimizer
        self.get_optimizer()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                        step_size=self.args.lr_step, 
                                        gamma=self.args.gamma)

        
        self.criterion = self.get_loss()
        self.lr_scheduler_net = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                                        self.optimizer_net, 
                                                                        mode="min", 
                                                                        patience=self.args.patience, 
                                                                        factor=self.args.factor)


        # if you wanna resume (load from checkpoint)
        self.strat_epoch = 1
        #self.resume_from_checkpoint()


    def get_loss(self):
        if self.args.model_type == "arcface":
            if self.args.loss == "focal_loss":
                criterion = losses.FocalLoss(gamma=2)

            elif self.args.loss == "cross_entropy":
                criterion = torch.nn.CrossEntropyLoss()

        elif self.args.model_type == "adaface":
            criterion = torch.nn.CrossEntropyLoss()
        return criterion


    def get_margin(self):
        if self.args.model_type == "arcface":
            metric_fc = metrics.ArcMarginProduct(self.args.fusion_final_dim, 
                                                self.args.num_classes, 
                                                s=30, 
                                                m=0.5, 
                                                easy_margin=self.args.easy_margin)

        elif self.args.model_type == "adaface":
            metric_fc = AdaFace(embedding_size = self.args.fusion_final_dim,
                                classnum = self.args.num_classes)

        metric_fc.to(self.args.device)
        metric_fc = torch.nn.DataParallel(metric_fc, device_ids=self.args.gpu_id)
        return metric_fc



    def get_optimizer(self):
        params = [{"params": self.metric_fc.parameters(), 
                   "lr" : self.args.lr_image_train, 
                   "weight_decay" : self.args.weight_decay}]
        
        params_en = [{"params": self.text_encoder.parameters(), "lr" : 5e-4}]
        params_net = [{"params": itertools.chain(self.fusion_net.parameters()), 
                       "lr": self.args.lr_head}]

        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(params)

        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params)

        self.optimizer_en = torch.optim.AdamW(params_en,  betas=(0.9, 0.999), weight_decay=0.01)
        self.optimizer_net = torch.optim.Adam(params_net,  betas=(0.5, 0.999)) 


    def train(self):

        device = self.args.device
        self.fusion_net.train()
        self.text_encoder.train()

        loop = tqdm(total=len(self.train_dl))
        total_loss = 0

        for data in self.train_dl:
        
            imgs, words_emb, sent_emb, keys, label, cap_lens = \
                    prepare_train_data(data, self.text_encoder)
            
            # load cuda
            imgs = imgs.to(device).requires_grad_()
            words_emb = words_emb.to(device).requires_grad_()
            sent_emb = sent_emb.to(device).requires_grad_()
            label = label.to(device)
            
            if self.args.model_type == "arcface":
                global_feats, local_feats = get_features(self.model, imgs)

            elif self.args.model_type == "adaface":
                global_feats, local_feats, norm = get_features_adaface(self.model, imgs)


            if self.args.fusion_type == "linear":
                output = self.fusion_net(global_feats, sent_emb)

            elif self.args.fusion_type == "concat":
                output = self.fusion_net(global_feats, sent_emb)

            elif self.args.fusion_type == "concat_attention":
                output = self.fusion_net(global_feats, sent_emb)

            elif self.args.fusion_type == "cross_attention":
                output = self.fusion_net(local_feats, words_emb)

            
            if self.args.model_type == "arcface":
                output = self.metric_fc(output, label)
            elif self.args.model_type == "adaface":
                output = self.metric_fc(output, norm, label)

            loss = self.criterion(output, label)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            if self.args.current_epoch < 11: self.optimizer_en.zero_grad()
            self.optimizer_net.zero_grad()
            loss.backward()
            
            self.optimizer.step()
            if self.args.current_epoch < 11: self.optimizer_en.step()
            self.optimizer_net.step()
            self.lr_scheduler_en.step()

            # update loop information
            loop.update(1)
            loop.set_description(f'Training Epoch [{self.args.current_epoch}/{self.args.max_epoch}]')
            loop.set_postfix()

            del global_feats, local_feats, loss 
            
        loop.close()
        self.scheduler.step()
        self.lr_scheduler_net.step(total_loss)
        str_loss = "CE loss {:0.4f}".format(total_loss / (len(self.train_dl) * self.args.batch_size))
        print(str_loss)


    def save_models(self):
        save_dir = os.path.join(self.args.checkpoints_path, 
                                self.args.dataset_name, 
                                self.args.CONFIG_NAME, 
                                self.args.en_type + "_" + self.args.model_type + "_" + self.args.fusion_type,
                                today.strftime("%m-%d-%y"))
        mkdir_p(save_dir)

        name = '%s_model_%s_%d.pth' % (self.args.model_type, self.args.fusion_type, self.args.current_epoch)
        state_path = os.path.join(save_dir, name)
        state = {'model': {'net': self.fusion_net.state_dict(), 
                        'metric_fc': self.metric_fc.state_dict()},
                'optimizer': {'optimizer': self.optimizer.state_dict()}}
        
        torch.save(state, state_path)

        checkpoint_text_en = {
            'model': self.text_encoder.state_dict(),
        }

        torch.save(checkpoint_text_en, '%s/%s_model_%s_%d.pth' % 
                    (save_dir, self.args.en_type, self.args.fusion_type, self.args.current_epoch))


    def main(self):
        #pprint.pprint(self.args)
        print("Start Training")
        for epoch in range(self.strat_epoch, self.args.max_epoch + 1):
            torch.cuda.empty_cache()
            self.args.current_epoch = epoch

            print('Reset scheduler')
            self.lr_scheduler_en = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_en, 
                                                                              T_max=1000, 
                                                                              eta_min=1e-4)
            
            self.train()
            if epoch % self.args.save_interval==0:
                self.text_head = None
                self.save_models()

            if ((self.args.do_test == True) and (epoch % self.args.test_interval == 0)):
                print("\nLet's test the model")
                self.text_head = None 
                test(self.valid_dl, self.model, self.fusion_net, self.text_encoder, self.text_head, self.args)


if __name__ == "__main__":
    args = merge_args_yaml(parse_args())

    # set seed
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)
    args.device = torch.device("cuda")
    
    Fusion(args).main()