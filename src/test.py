import sys
import os.path as osp
import random
import argparse
import numpy as np
import torch


ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils.utils import merge_args_yaml, load_fusion_net
from utils.prepare import prepare_dataloader, prepare_model, prepare_text_encoder, prepare_fusion_net, prepare_adaface
from utils.modules import test


def parse_args():
    # Training settings
    print("loading celeba.yml")
    parser = argparse.ArgumentParser(description='CGFG')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='./cfg/celeba.yml',
                        help='optional config file')
    parser.add_argument('--train', type=bool, default=False, help='if train model')
    args = parser.parse_args()
    return args


def main(args):
    test_dl, test_ds = prepare_dataloader(args, split="test", transform=None)

    if args.using_BERT == False:
        args.vocab_size = test_ds.n_words 
    
    if args.model_type == "arcface":
            model = prepare_model(args)

    elif args.model_type == "adaface":
            model = prepare_adaface(args)

    
    # load from checkpoint
    net = prepare_fusion_net(args)
    
    for i in [80]:
        args.text_encoder_path = "./checkpoints/celeba/Pretrain/BiLSTM/adaface/best_adaface_text_encoder_bert_%d.pth" % i
        text_encoder, text_head = prepare_text_encoder(args, test=True)

        if args.fusion_type != "concat":
            print("loading checkpoint; epoch: ", i)
            load_path = "./checkpoints/celeba/Fusion/final_cross_att/bert_cross_attention_epoch_%d.pth" % i
            net = load_fusion_net(net, load_path) 
        
        #pprint.pprint(args)
        print("Start Testing")
        args.is_roc = True   
        test(test_dl, model, net, text_encoder, text_head, args)
    



if __name__ == "__main__":
    args = merge_args_yaml(parse_args())

    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)
    #torch.cuda.set_device(args.gpu_id)
    args.device = torch.device("cuda")

    main(args)