import sys
import os.path as osp
import random
import argparse
import numpy as np
import pprint
import torch


ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils.utils import mkdir_p, merge_args_yaml
from utils.utils import load_models
from utils.perpare import get_test_dataloader, prepare_models
from utils.modules import test 


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='./cfg/birds.yml',
                        help='optional config file')
    parser.add_argument('--train', type=bool, default=False, help='if train model')
    args = parser.parse_args()
    return args


def main(args):
    # prepare dataloader, models, data
    args.model_save_file = osp.join(ROOT_PATH, 'saved_models', str(args.CONFIG_NAME))
    mkdir_p(args.model_save_file)

    test_dl, test_ds = get_test_dataloader(args)

    args.vocab_size = test_ds.n_words
    image_encoder, text_encoder, model, netG = prepare_models(args)
    
    # load from checkpoint
    path = osp.join(args.resume_model_path, 'state_epoch_%03d.pth'%(args.resume_epoch))
    netG = load_models(netG, path)

    #pprint.pprint(args)
    print("Start Testing")
    test(test_dl, model, netG.module.fc1, text_encoder, args)


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