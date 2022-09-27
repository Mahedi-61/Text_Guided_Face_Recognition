import sys
import os.path as osp
import random
import argparse
import numpy as np
import pprint
import torch
from torch import nn 

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils.utils import mkdir_p, merge_args_yaml
from utils.perpare import get_test_dataloader, prepare_models
from utils.modules import * 

def test(test_dl, model, args):
    device = args.device
    model = model.eval()
    preds = []
    labels = []

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

    best_acc, best_th = cal_accuracy(preds, labels)
    print("accuracy: ", best_acc)
    calculate_scores(preds, labels)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='ImageRec')
    parser.add_argument('--cfg', 
                        dest='cfg_file', type=str, 
                        default='./cfg/celeba.yml',
                        help='optional config file')
    parser.add_argument('--train', type=bool, default=False, help='if train model')
    args = parser.parse_args()
    return args


def main(args):
    # prepare dataloader, models, data
    args.model_save_file = osp.join(ROOT_PATH, 'saved_models', str(args.dataset_name))
    mkdir_p(args.model_save_file)

    test_dl, test_ds = get_test_dataloader(args)
    args.vocab_size = test_ds.n_words
    image_encoder, text_encoder, model, netG = prepare_models(args)
  
    pprint.pprint(args)
    print("Start Testing")
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