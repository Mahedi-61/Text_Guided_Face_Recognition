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
from utils.prepare import (prepare_dataloader, 
                           prepare_text_encoder, 
                           prepare_adaface, prepare_arcface, prepare_image_head,
                           prepare_train_data)

from utils.modules import test
from models import metrics, losses
from models.fusion_nets import LinearFusion
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
        print("Loading training and valid data ...")
        del train_ds, valid_ds 

        self.text_encoder, self.text_head = prepare_text_encoder(self.args)
        
        if self.args.model_type == "arcface":
            self.image_encoder = prepare_arcface(self.args) #cuda + parallel + grd. false + eval
            
        elif self.args.model_type == "adaface":
            self.image_encoder = prepare_adaface(self.args)

        self.image_head = prepare_image_head(self.args)
        if args.fusion_type == "linear":
            self.fusion_net = LinearFusion(args)

        self.fusion_net = torch.nn.DataParallel(self.fusion_net, 
                                                device_ids=args.gpu_id).to(args.device)
        self.metric_fc = self.get_margin()


        # prepare loss & optimizer
        self.get_optimizer()
        
        self.criterion = self.get_loss()
        self.lr_scheduler_en = torch.optim.lr_scheduler.StepLR(self.optimizer_en, 
                                        step_size=10, 
                                        gamma=0.8)

        self.lr_scheduler_cls = torch.optim.lr_scheduler.StepLR(self.optimizer_cls, 
                                        step_size=5, 
                                        gamma=0.6)
    
        self.lr_scheduler_net = torch.optim.lr_scheduler.StepLR(self.optimizer_net, 
                                        step_size=5, 
                                        gamma=0.97)
        self.strat_epoch = 1
        #self.resume_from_checkpoint()


    def get_loss(self):
        if self.args.model_type == "arcface":
            criterion = losses.FocalLoss(gamma=2)

        elif self.args.model_type == "adaface":
            criterion = torch.nn.CrossEntropyLoss()
        return criterion


    def get_margin(self):
        metric_fc = metrics.ArcMarginProduct(self.args.fusion_final_dim, 
                                            self.args.num_classes, 
                                            s=30, 
                                            m=0.5, 
                                            easy_margin=self.args.easy_margin)

        """
        if self.args.model_type == "arcface":
        elif self.args.model_type == "adaface":
            metric_fc = AdaFace(embedding_size = self.args.fusion_final_dim,
                                classnum = self.args.num_classes)
        """
    
        metric_fc.to(self.args.device)
        metric_fc = torch.nn.DataParallel(metric_fc, device_ids=self.args.gpu_id)
        return metric_fc



    def get_optimizer(self):
        params_cls = [{"params": self.metric_fc.parameters(), 
                   "lr" : self.args.lr_image_train, 
                   "weight_decay" : self.args.weight_decay}]
        
        params_en = [{"params": self.text_encoder.parameters(), "lr" : 3e-4}]
        params_net = [{"params": itertools.chain(
                    self.image_head.parameters(), 
                    self.fusion_net.parameters()), 
                    "lr": self.args.lr_head}]

        self.optimizer_cls = torch.optim.SGD(params_cls)
        self.optimizer_en = torch.optim.Adam(params_en,  
                                             betas=(0.9, 0.999), 
                                             weight_decay=0.01)
        
        self.optimizer_net = torch.optim.Adam(params_net,  
                                              betas=(0.5, 0.999)) 


    def get_fusion_output(self, sent_emb, words_emb, global_feats, local_feats):
        if self.args.fusion_type == "linear":
            output = self.fusion_net(global_feats, sent_emb)

        elif self.args.fusion_type == "concat":
            output = self.fusion_net (global_feats, sent_emb)

        elif self.args.fusion_type == "fcfm":
            output = self.fusion_net (local_feats, words_emb, global_feats, sent_emb)

        return output 
    

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
            
            if self.args.model_type == "adaface":
                img_feats, local_feats, norm = self.image_encoder(imgs)
            else:
                img_feats, local_feats = self.image_encoder(imgs)

            img_feats, local_feats = self.image_head(img_feats, local_feats)
            output = self.get_fusion_output(sent_emb, words_emb, img_feats, local_feats)

            if self.args.model_type == "arcface":
                output = self.metric_fc(output, label)
            elif self.args.model_type == "adaface":
                output = self.metric_fc(output, label) #norm, 

            self.optimizer_cls.zero_grad()
            self.optimizer_en.zero_grad()
            self.optimizer_net.zero_grad()

            loss = self.criterion(output, label)
            loss.backward()
            total_loss += loss.item()
        
            self.optimizer_cls.step() 
            self.optimizer_en.step()
            self.optimizer_net.step()

            # update loop information
            loop.update(1)
            loop.set_description(f'Training Epoch [{self.args.current_epoch}/{self.args.max_epoch}]')
            loop.set_postfix()

            
        loop.close()
        str_loss = "CE loss {:0.4f}".format(total_loss / (len(self.train_dl) * self.args.batch_size))
        print(str_loss)



    def save_models(self):
        save_dir = os.path.join(self.args.checkpoints_path, 
                                self.args.dataset_name, 
                                self.args.CONFIG_NAME, 
                                "BiLSTM_" + self.args.model_type,
                                self.args.fusion_type,
                                today.strftime("%m-%d-%y-%H:%M"))
        mkdir_p(save_dir)

        name = 'fusion_%s_%s_%d.pth' % (args.fusion_type,
                                        args.model_type, 
                                        self.args.current_epoch )
        state_path = os.path.join(save_dir, name)

        state = {'net': self.fusion_net.state_dict(), 
                 'image_head': self.image_head.state_dict(),
                }
        
        torch.save(state, state_path)
        checkpoint_text_en = {
            'model': self.text_encoder.state_dict()
        }

        torch.save(checkpoint_text_en, '%s/encoder_%s_%s_%d.pth' % 
                    (save_dir, args.en_type, args.fusion_type, self.args.current_epoch ))



    def main(self):
        #pprint.pprint(self.args)
        print("Start Training")
        for epoch in range(self.strat_epoch, self.args.max_epoch + 1):
            torch.cuda.empty_cache()
            self.args.current_epoch = epoch

            self.train()
            self.lr_scheduler_cls.step()
            self.lr_scheduler_en.step()
            self.lr_scheduler_net.step()

            if epoch % self.args.save_interval==0:
                self.text_head = None
                self.save_models()

            if ((self.args.do_test == True) and (epoch % self.args.test_interval == 0)):
                print("\nLet's test the model")
                self.text_head = None 
                test(self.valid_dl, 
                    self.image_encoder, self.image_head,  
                    self.fusion_net, 
                    self.text_encoder, self.text_head, 
                    self.args)


if __name__ == "__main__":
    args = merge_args_yaml(parse_args())

    # set seed
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)
    args.device = torch.device("cuda")
    
    Fusion(args).main()