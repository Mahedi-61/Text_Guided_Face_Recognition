import sys
import os.path as osp
import random
import argparse
import numpy as np
import pprint
import torch
from torch import nn 
from tqdm import tqdm 

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils.utils import mkdir_p, merge_args_yaml
from utils.prepare import get_test_dataloader, prepare_models
from utils.modules import * 


def test(test_dl, model, args):
    device = args.device
    model = model.eval()
    preds = []
    labels = []

    loop = tqdm(total = len(test_dl))
    for step, data in enumerate(test_dl, 0):
        img1, img2, cap1, cap2, cap_len1, cap_len2, pair_label  = data
        img1 = img1.to(device).requires_grad_()
        img2 = img2.to(device).requires_grad_()
        pair_label = pair_label.to(device)

        out1 = get_features(model, img1)
        out2 = get_features(model, img2)

        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        pred = cosine_sim(out1, out2)
        preds += pred.data.cpu().tolist()
        labels += pair_label.data.cpu().tolist()

        # update loop information
        loop.update(1)
        loop.set_description(f'Testing')
        loop.set_postfix()
        
    print(out2.shape)
    loop.close()
    calculate_scores(preds, labels)
    #best_acc, best_th = cal_accuracy(preds, labels)
    #print("accuracy: %0.4f thereshold %0.4f" % (best_acc, best_th))
    


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='ImageRec')
    parser.add_argument('--cfg', 
                        dest='cfg_file', type=str, 
                        default='./cfg/FE_celeba.yml',
                        help='optional config file')
    parser.add_argument('--train', type=bool, default=False, help='if train model')
    args = parser.parse_args()
    return args


def main(args):
    test_dl, test_ds = get_test_dataloader(args)
    args.vocab_size = test_ds.n_words
    print("loading models ...")
    model, netG = prepare_models(args)
  
    #pprint.pprint(args)
    print("start testing ...")
    print("test file: ", args.test_pair_list)
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