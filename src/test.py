import sys
import os.path as osp
import random
import argparse
import numpy as np
import pprint
import torch


ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils.utils import merge_args_yaml
from utils.utils import load_model
from utils.prepare import prepare_dataloader, prepare_models, prepare_text_encoder
from utils.modules import test 


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='./cfg/celeba.yml',
                        help='optional config file')
    parser.add_argument('--train', type=bool, default=False, help='if train model')
    args = parser.parse_args()
    return args


def main(args):
    test_dl, test_ds = prepare_dataloader(args, split="test", transform=None)

    args.vocab_size = test_ds.n_words
    text_encoder = prepare_text_encoder(args)
    model, net = prepare_models(args)

    
    # load from checkpoint
    print("loading checkpoint; epoch: ", args.resume_epoch)
    net = load_model(net, args.resume_model_path)

    #pprint.pprint(args)
    print("Start Testing")
    test(test_dl, model, net, text_encoder, args)


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