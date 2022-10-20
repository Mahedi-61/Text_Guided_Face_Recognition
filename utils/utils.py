import os
import errno
from tabnanny import check
import numpy as np
import torch
import pickle
import yaml
from easydict import EasyDict as edict
import pprint
import datetime
import dateutil.tz
from PIL import Image


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


def save_models(net, metric_fc, optG, epoch, args):
    save_dir = os.path.join(args.checkpoints_path, 
                            args.dataset_name, 
                            args.CONFIG_NAME)
    mkdir_p(save_dir)

    name = '%s/state_epoch_%s_%s_%03d.pth' % (args.model_save_file, args.en_type, args.fusion_type, epoch)
    state_path = os.path.join(save_dir, name)
    state = {'model': {'net': net.state_dict(), 'metric_fc': metric_fc.state_dict()},
            'optimizer': {'optimizer': optG.state_dict()}}
    torch.save(state, state_path)


def load_models(net, metric_fc, optim, path):
    print("loading full tgfr model .....")
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    net = load_model_weights(net, checkpoint['model']['net'])
    metric_fc = load_model_weights(metric_fc, checkpoint['model']['metric_fc'])
    optim = optim.load_state_dict(checkpoint['optimizer']['optimizer'])
    return net, metric_fc, optim 


def load_fusion_net(net, path):
    print("loading full tgfr model .....")
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    net = load_model_weights(net, checkpoint['model']["net"])
    return net


def load_pretrained_arch_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint) 
    return model 



"""
###########  GEN  #############
def get_tokenizer():
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer


def tokenize(wordtoix, text_filepath):
    '''generate images from example sentences'''
    tokenizer = get_tokenizer()
    filepath = text_filepath
    with open(filepath, "r") as f:
        sentences = f.read().split('\n')
        # a list of indices for a sentence
        captions = []
        cap_lens = []
        new_sent = []
        for sent in sentences:
            if len(sent) == 0:
                continue
            sent = sent.replace("\ufffd\ufffd", " ")
            tokens = tokenizer.tokenize(sent.lower())
            if len(tokens) == 0:
                print('sent', sent)
                continue
            rev = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0 and t in wordtoix:
                    rev.append(wordtoix[t])
            captions.append(rev)
            cap_lens.append(len(rev))
            new_sent.append(sent)
        return captions, cap_lens, new_sent


def sort_example_captions(captions, cap_lens, device):
    max_len = np.max(cap_lens)
    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    captions = torch.from_numpy(cap_array).to(device)
    cap_lens = torch.from_numpy(cap_lens).to(device)
    return captions, cap_lens, sorted_indices



def encode_tokens(text_encoder, caption, cap_lens):
    # encode text
    with torch.no_grad():
        if hasattr(text_encoder, 'module'):
            hidden = text_encoder.module.init_hidden(caption.size(0))
        else:
            hidden = text_encoder.init_hidden(caption.size(0))
        words_embs, sent_emb = text_encoder(caption, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    return sent_emb, words_embs 


def sort_sents(captions, caption_lens, device):
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(caption_lens, 0, True)
    captions = captions[sorted_cap_indices].squeeze()
    captions = captions.to(device)
    sorted_cap_lens = sorted_cap_lens.to(device)
    return captions, sorted_cap_lens, sorted_cap_indices


def rm_sort(caption, sorted_cap_idxs):
    non_sort_cap = torch.empty_like(caption)
    for idx, sort in enumerate(sorted_cap_idxs):
        non_sort_cap[sort] = caption[idx]
    return non_sort_cap


def save_img(img, path):
    im = img.data.cpu().numpy()
    # [-1, 1] --> [0, 255]
    im = (im + 1.0) * 127.5
    im = im.astype(np.uint8)
    im = np.transpose(im, (1, 2, 0))
    im = Image.fromarray(im)
    im.save(path)
"""