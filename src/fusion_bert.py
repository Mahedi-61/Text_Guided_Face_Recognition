import os 
import sys
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
                           prepare_arcface, prepare_adaface, prepare_image_head,
                           prepare_train_data_for_Bert)

from utils.modules import test
from models.fusion_nets import LinearFusion, Working
from utils.utils import load_models   
from models import metrics, losses
from datetime import datetime 
today = datetime.now() 


def parse_args():
    # Training settings
    cfg_file = "fusion_bert.yml"
    print("Loading: ", cfg_file)
    parser = argparse.ArgumentParser(description='Fusion')
    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/%s' %cfg_file,
                        help='optional config file')
    args = parser.parse_args()
    return args


class Fusion:
    def __init__(self, args):

        self.args = args 
        self.train_dl, train_ds = prepare_dataloader(self.args, split="train", transform=None)
        self.valid_dl, valid_ds = prepare_dataloader(self.args, split="valid", transform=None)
        print("Loading training and valid data ...")
        del train_ds, valid_ds 

        # preapare model
        self.text_encoder, self.text_head = prepare_text_encoder(self.args)
        
        if self.args.model_type == "arcface":
            self.image_encoder = prepare_arcface(self.args) 
            
        elif self.args.model_type == "adaface":
            self.image_encoder = prepare_adaface(self.args)

        self.image_head = prepare_image_head(self.args)

        if args.fusion_type == "linear":
            self.fusion_net = LinearFusion(args)

        elif args.fusion_type == "fcfm": 
            self.fusion_net = Working(channel_dim = args.aux_feat_dim_per_granularity)

        self.fusion_net = torch.nn.DataParallel(self.fusion_net, 
                                                device_ids=args.gpu_id).to(args.device)
        self.metric_fc = self.get_margin()

        # prepare loss & optimizer
        self.criterion = self.get_loss()
        self.optimizer_cls, self.optimizer_en, self.optimizer_head = self.get_optimizer()

        self.lr_scheduler_en = torch.optim.lr_scheduler.StepLR(self.optimizer_en, 
                                        step_size=10, 
                                        gamma=0.8)

        self.lr_scheduler_cls = torch.optim.lr_scheduler.StepLR(self.optimizer_cls, 
                                        step_size=5, 
                                        gamma=0.6)
    
        self.lr_scheduler_head = torch.optim.lr_scheduler.StepLR(self.optimizer_head, 
                                        step_size=5, 
                                        gamma=0.97)

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
        metric_fc = metrics.ArcMarginProduct(self.args.fusion_final_dim, 
                                            self.args.num_classes, 
                                            s=30, 
                                            m=0.5, 
                                            easy_margin=self.args.easy_margin)


        metric_fc.to(self.args.device)
        metric_fc = torch.nn.DataParallel(metric_fc, device_ids=self.args.gpu_id)
        return metric_fc


    def get_optimizer(self):
        params_cls = [{"params": self.metric_fc.parameters(), 
                      "lr" : self.args.lr_image_train, 
                      "weight_decay" : self.args.weight_decay}]
        
        params_en = [{"params": self.text_encoder.parameters()}]

        params_head = [{"params": itertools.chain(self.text_head.parameters(),
                                                  self.image_head.parameters(), 
                                                  self.fusion_net.parameters())
                        }]

        optimizer_cls = torch.optim.SGD(params_cls)

        optimizer_en = torch.optim.Adam(params_en,  
                                        betas=(0.9, 0.999), 
                                        weight_decay=0.01,
                                        lr = 1e-5)
        
        optimizer_head = torch.optim.Adam(params_head, 
                                          weight_decay=5e-5, 
                                          lr=self.args.lr_head)
        
        return optimizer_cls, optimizer_en, optimizer_head

 
    def get_fusion_output(self, sent_emb, words_emb, global_feats, local_feats):

        if self.args.fusion_type == "linear":
            output = self.fusion_net(global_feats, sent_emb)

        elif self.args.fusion_type == "concat":
            output = self.fusion_net (global_feats, sent_emb)

        elif self.args.fusion_type == "fcfm":
            output = self.fusion_net (local_feats, words_emb, global_feats, sent_emb)

        return output 


    def resume_from_checkpoint(self):
        if self.args.resume_epoch!=1:
            print("loading checkpoint; epoch: ", self.args.resume_epoch)
            strat_epoch = self.args.resume_epoch+1
            self.fusion_net, self.metric_fc, self.optimizer= load_models(self.fusion_net, 
                            self.metric_fc, self.optimizer, self.args.resume_model_path)


    def save_models(self):
        save_dir = os.path.join(self.args.checkpoints_path, 
                                self.args.dataset_name, 
                                self.args.CONFIG_NAME, 
                                self.args.en_type + "_" + self.args.model_type,
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
            'model': self.text_encoder.state_dict(),
            'head': self.text_head.state_dict()
        }

        torch.save(checkpoint_text_en, '%s/encoder_%s_%s_%d.pth' % 
                    (save_dir, args.en_type, args.fusion_type, self.args.current_epoch ))


    def train(self):
        device = self.args.device
        self.fusion_net.train()
        self.metric_fc.train()
        self.image_head.train() 
        self.text_encoder.train()
        self.text_head.train()

        loop = tqdm(total=len(self.train_dl))
        total_loss = 0

        for data in self.train_dl:
            imgs, caps, masks, keys, class_ids = data 
            words_emb, sent_emb = prepare_train_data_for_Bert((caps, masks), self.text_encoder, self.text_head)
            cap_lens = None 

            # load cuda
            words_emb = words_emb.to(device).requires_grad_()
            sent_emb = sent_emb.to(device).requires_grad_()
            label = class_ids.to(device)

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
            self.optimizer_head.zero_grad()

            loss = self.criterion(output, label)
            loss.backward()
            total_loss += loss.item()

            self.optimizer_cls.step()
            self.optimizer_en.step()
            self.optimizer_head.step()

            # update loop information
            loop.update(1)
            loop.set_description(f'Training Epoch [{self.args.current_epoch}/{self.args.max_epoch}]')
            loop.set_postfix()

        loop.close()

        str_loss = " | loss {:0.5f}".format(total_loss / (len(self.train_dl) * self.args.batch_size))
        print(str_loss)
    

    def main(self):
        print("Start Training")

        for epoch in range(self.strat_epoch, self.args.max_epoch + 1):
            torch.cuda.empty_cache()
            self.args.current_epoch = epoch

            self.train()
            self.lr_scheduler_cls.step()
            self.lr_scheduler_en.step()
            self.lr_scheduler_head.step()
        
            # save
            if epoch % self.args.save_interval==0:
                self.save_models()
            
            if (epoch > 20):
                if ((self.args.do_test == True) and (epoch % self.args.test_interval == 0)):
                    print("\nLet's test the model")
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