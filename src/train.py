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
from utils.utils import load_model_opt, save_models, load_models
from utils.perpare import get_train_dataloader, get_test_dataloader, prepare_models
from utils.modules import train, test  


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/celeba.yml',
                        help='optional config file')
    parser.add_argument('--train', type=bool, default=True, help='if train model')
    args = parser.parse_args()
    return args


def main(args):
    # prepare dataloader, models, data
    args.model_save_file = osp.join(ROOT_PATH, 'saved_models', str(args.dataset_name))
    mkdir_p(args.model_save_file)

    train_dl, train_ds = get_train_dataloader(args)
    test_dl, test_ds = get_test_dataloader(args)

    args.vocab_size = train_ds.n_words
    image_encoder, text_encoder, model, netG = prepare_models(args)

    optimizerG = torch.optim.Adam(netG.parameters(), 
                                 lr=0.0009, 
                                 betas=(0.0, 0.9))
                                 
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizerG, 
                                    lr_lambda=lambda epoch: args.multi_lr_step)
    
    # load from checkpoint
    strat_epoch = 1
    if args.resume_epoch!=1:
        strat_epoch = args.resume_epoch+1
        path = osp.join(args.resume_model_path, 'state_epoch_%03d.pth'%(args.resume_epoch))
        netG, optimizerG = load_model_opt(netG, optimizerG, path)

    #pprint.pprint(args)
    print("Start Training")
    for epoch in range(strat_epoch, args.max_epoch, 1):
        torch.cuda.empty_cache()
        args.current_epoch = epoch 
        train(train_dl, model, netG, text_encoder, optimizerG, args)
        
        # save
        if epoch % args.save_interval==0:
            save_models(netG, optimizerG, epoch, args.model_save_file)
            scheduler.step()
            print("learning rate: ", optimizerG.param_groups[0]['lr'])
        
        if ((args.do_test == True) and (epoch % args.test_interval == 0)):
            print("Let's test the model")
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
    args.device = torch.device("cuda")

    main(args)

#python3 -m torch.distributed.launch --nproc_per_node=2 src/train.py