import sys
from models import iresnet
import os
import torch.nn.functional as F
import torch.nn as nn
import torch


def load_features(arch):
    if arch == 'iresnet34':
        features = iresnet.iresnet34(pretrained=False, progress=True)

    elif arch == 'iresnet18':
        features = iresnet.iresnet18(pretrained=False, progress=True)

    elif arch == 'iresnet50':
        features = iresnet.iresnet50(pretrained=False, progress=True)

    elif arch == 'iresnet100':
        features = iresnet.iresnet100(pretrained=False, progress=True)

    else:
        raise ValueError()
    return features


class NetworkBuilder(nn.Module):
    def __init__(self, arch):
        super(NetworkBuilder, self).__init__()
        self.features = load_features(arch)

    def forward(self, input):
        # add Fp, a pose feature
        gl_featus, lc_feats = self.features(input)
        return gl_featus, lc_feats 

"""
def load_dict_inf(args, model):
    if os.path.isfile(args.resume):
        cprint('=> loading pth from {} ...'.format(args.resume))
        if args.cpu_mode:
            checkpoint = torch.load(args.resume, map_location=torch.device("cpu"))
        else:
            checkpoint = torch.load(args.resume)
        _state_dict = clean_dict_inf(model, checkpoint['state_dict'])
        model_dict = model.state_dict()
        model_dict.update(_state_dict)
        model.load_state_dict(model_dict)
        # delete to release more space
        del checkpoint
        del _state_dict
    else:
        sys.exit("=> No checkpoint found at '{}'".format(args.resume))
    return model


def clean_dict_inf(model, state_dict):
    _state_dict = OrderedDict()
    for k, v in state_dict.items():
        # # assert k[0:1] == 'features.module.'
        new_k = 'features.'+'.'.join(k.split('.')[2:])
        if new_k in model.state_dict().keys() and \
           v.size() == model.state_dict()[new_k].size():
            _state_dict[new_k] = v
        # assert k[0:1] == 'module.features.'
        new_kk = '.'.join(k.split('.')[1:])
        if new_kk in model.state_dict().keys() and \
           v.size() == model.state_dict()[new_kk].size():
            _state_dict[new_kk] = v
    num_model = len(model.state_dict().keys())
    num_ckpt = len(_state_dict.keys())
    if num_model != num_ckpt:
        sys.exit("=> Not all weights loaded, model params: {}, loaded params: {}".format(
            num_model, num_ckpt))
    return _state_dict


def builder_inf(args):
    model = NetworkBuilder_inf(args)
    # Used to run inference
    model = load_dict_inf(args, model)
    return model
"""


if __name__ == "__main__":
    from easydict import EasyDict
    args = EasyDict() 
    args.arch = "iresnet18"
    args.embedding_size = 512 
    resnet = NetworkBuilder(args)

    x = torch.randn(32, 3, 112, 112).cuda()
    resnet = torch.nn.DataParallel(resnet, device_ids=[0]).cuda()


    mag_dict = torch.load("magface_iresnet18_casia_dp.pth")
    del mag_dict["state_dict"]["module.fc.weight"]
    resnet.load_state_dict(mag_dict["state_dict"])
    y, lfeat = resnet(x)
    print(y.shape)
    print(lfeat.shape)  