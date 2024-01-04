import sys
import os.path as osp
import random
import argparse
import numpy as np
import torch


ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)

from utils.utils import merge_args_yaml
from utils.prepare import (prepare_dataloader, 
                           prepare_arcface, prepare_adaface, prepare_image_head,
                           prepare_text_encoder, 
                           prepare_fusion_net)
from utils.modules import test


def parse_args():
    # Training settings
    print("loading celeba.yml")
    cfg_file = "test.yml"
    parser = argparse.ArgumentParser(description='Testing TGFR model')
    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/%s' % cfg_file,
                        help='optional config file')
    args = parser.parse_args()
    return args


class Test:
    def __init__(self, args):
        self.args = args 
        self.test_dl, test_ds = prepare_dataloader(args, split="test", transform=None)
        
        if self.args.en_type == "LSTM":
            self.args.vocab_size = test_ds.n_words 

        # preapare model
        self.text_encoder, self.text_head = prepare_text_encoder(self.args)
        
        if self.args.model_type == "arcface":
            self.image_encoder = prepare_arcface(self.args) 
            
        elif self.args.model_type == "adaface":
            self.image_encoder = prepare_adaface(self.args)

        self.image_head = prepare_image_head(self.args)

        if args.fusion_type != "concat":
            self.fusion_net = prepare_fusion_net(self.args) 
        else:
            self.fusion_net = None 


    def main(self):
        #pprint.pprint(self.args)
        print("\nLet's test the model")
        test(self.test_dl, 
            self.image_encoder, self.image_head,  
            self.fusion_net, 
            self.text_encoder, self.text_head, 
            self.args)
    

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

    Test(args).main()