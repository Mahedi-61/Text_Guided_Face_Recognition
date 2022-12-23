import sys
import os.path as osp
import random
import argparse
import numpy as np
import torch
from torch import nn 
from tqdm import tqdm 
ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)

from utils.utils import   merge_args_yaml
from utils.prepare import prepare_dataloader, prepare_model, prepare_adaface
from utils.modules import * 


def test(test_dl, model, args):
    device = args.device
    model = model.eval()
    preds = []
    labels = []
    loop = tqdm(total = len(test_dl))

    for step, data in enumerate(test_dl, 0):
        img1, img2, cap1, cap2, cap_len1, cap_len2, pair_label = data
        
        img1 = img1.to(device).requires_grad_()
        img2 = img2.to(device).requires_grad_()
        pair_label = pair_label.to(device)


        # get global and local image features from arc face model
        global_feat1,  local_feat1 = get_features(model, img1)
        global_feat2,  local_feat2 = get_features(model, img2)

        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        pred = cosine_sim(global_feat1, global_feat2)
        preds += pred.data.cpu().tolist()
        labels += pair_label.data.cpu().tolist()

        # update loop information
        loop.update(1)
        loop.set_postfix()

    loop.close()
    best_acc, best_th = cal_accuracy(preds, labels)
    calculate_scores(preds, labels, args)
    print("accuracy: %0.4f; threshold %0.4f" %(best_acc, best_th))


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='CGFG')
    parser.add_argument('--cfg', 
                        dest='cfg_file', type=str, 
                        default='./cfg/celeba.yml',
                        help='optional config file')
    parser.add_argument('--train', type=bool, default=False, help='if train model')
    args = parser.parse_args()
    return args


def main(args):
    test_dl, test_ds = prepare_dataloader(args, split="test", transform=None)

    if args.using_BERT == False:
        args.vocab_size = test_ds.n_words


    print("loading models ...")
    if args.model_type == "adaface":
        model = prepare_adaface(args)
        
    elif args.model_type == "arcface":
        model = prepare_model(args)
  
    print("Start Testing")
    args.is_roc = True   
    test(test_dl, model, args)


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