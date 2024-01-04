import os
import errno
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
import datetime
import dateutil.tz
from datetime import date
today = date.today()

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# config
def get_time_stamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')  
    return timestamp


def load_yaml(filename):
    with open(filename, 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    return cfg


def merge_args_yaml(args):
    if args.cfg_file is not None:
        opt = vars(args)
        args = load_yaml(args.cfg_file)
        args.update(opt)
        args = edict(args)
    return args


def save_args(save_path, args):
    fp = open(save_path, 'w')
    fp.write(yaml.dump(args))
    fp.close()



def load_model_weights(model, weights, train=True):
    multi_gpus = True 
    if list(weights.keys())[0].find('module')==-1:
        pretrained_with_multi_gpu = False
    else:
        pretrained_with_multi_gpu = True
    if (multi_gpus==False) or (train==False):
        if pretrained_with_multi_gpu:
            state_dict = {
                key[7:]: value
                for key, value in weights.items()
            }
        else:
            state_dict = weights
    else:
        state_dict = weights
    model.load_state_dict(state_dict)
    return model



def load_models(net, metric_fc, optim, path):
    print("loading full tgfr model .....")
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    net = load_model_weights(net, checkpoint['model']['net'])
    metric_fc = load_model_weights(metric_fc, checkpoint['model']['metric_fc'])
    optim = optim.load_state_dict(checkpoint['optimizer']['optimizer'])
    return net, metric_fc, optim 


def load_fusion_net(net, path):
    print("loading fusion model .....")
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    net = load_model_weights(net, checkpoint["net"])
    return net
