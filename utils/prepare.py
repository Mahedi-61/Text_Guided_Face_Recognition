import torch
from utils.train_dataset import TrainDataset
from utils.test_dataset import TestDataset 

from utils.utils import load_model_weights, load_fusion_net
from models.models import RNNEncoder, TextHeading, TextEncoder, ImageHeading
from models.fusion_nets import (LinearFusion, Working, WordLevelCFA_LSTM)
from models import iresnet, net 
from models.network import NetworkBuilder
from utils.dataset_utils import *


###########   model   ############
def prepare_text_encoder(args):    
    if args.en_type == "BERT":
        text_encoder =  TextEncoder(args)
        text_encoder = torch.nn.DataParallel(text_encoder, device_ids=args.gpu_id).cuda()
        state_dict = torch.load(args.text_encoder_path)
        text_encoder.load_state_dict(state_dict['model'])

        text_head = TextHeading(args)
        text_head = torch.nn.DataParallel(text_head, device_ids=args.gpu_id).cuda()
        text_head.load_state_dict(state_dict['head'])
        del state_dict


    elif args.en_type == "LSTM":
        text_encoder = RNNEncoder(args, nhidden=args.embedding_dim)
        state_dict = torch.load(args.text_encoder_path, map_location='cpu')
        text_encoder = load_model_weights(text_encoder, state_dict["model"]) 
        
        text_encoder.cuda()
        text_head = None 

    print("loading text encoder weights: ", args.text_encoder_path)
    return text_encoder, text_head


def prepare_image_head(args):
    print("loading image encoder: ", args.image_encoder_path)
    head = ImageHeading(args)

    head = torch.nn.DataParallel(head, device_ids=args.gpu_id).cuda()
    state_dict = torch.load(args.image_encoder_path)
    head.load_state_dict(state_dict['image_head'])
    return head 



### model for ArcFace
def prepare_arcface(args):
    device = args.device
    model = iresnet.iresnet18(pretrained=False, progress=True)

    checkpoint = torch.load(args.weights_arcface)
    model.load_state_dict(checkpoint)

    model = torch.nn.DataParallel(model, device_ids=args.gpu_id).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print("loading pretrained arcface model")
    return model 


#### model for AdaFace 
def prepare_adaface(args):
    device = args.device
    architecture = "ir_18"

    model = net.build_model(architecture)    
    statedict = torch.load(args.weights_adaface)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print("loading pretrained adaface model")
    return model 


#### model for MagFace 
def prepare_magface(args):
    device = args.device
    resnet = NetworkBuilder(arch = "iresnet18")
    resnet = torch.nn.DataParallel(resnet, device_ids=args.gpu_id).to(device)

    mag_dict = torch.load(args.weights_magface)['state_dict']
    del mag_dict["module.fc.weight"]
    resnet.load_state_dict(mag_dict)
    
    for p in resnet.parameters():
        p.requires_grad = False
    resnet.eval()
    print("loading pretrained magface model")
    return resnet 



def prepare_fusion_net(args):
    # fusion models
    if args.fusion_type == "linear":
        net = LinearFusion(args)

    elif args.fusion_type == "fcfm":
        if args.en_type == "LSTM": 
            net = WordLevelCFA_LSTM(channel_dim = 256)

        elif args.en_type == "BERT":  
            net = Working(channel_dim = args.aux_feat_dim_per_granularity)

    net = torch.nn.DataParallel(net, device_ids=args.gpu_id).to(args.device)
    print("loading checkpoint; epoch: ", args.fusion_net_path)
    net = load_fusion_net(net, args.fusion_net_path) 
    
    return net



################ data ##############
def prepare_train_data(data, text_encoder):
    imgs, captions, caption_lens, keys, label = data
    captions, sorted_cap_lens, sorted_cap_idxs = sort_sents(captions, caption_lens)
    words_embs, sent_emb = encode_tokens(text_encoder, captions, sorted_cap_lens)
    sent_emb = rm_sort(sent_emb, sorted_cap_idxs)
    words_embs = rm_sort(words_embs, sorted_cap_idxs)
    return imgs, words_embs, sent_emb, keys, label, caption_lens



def prepare_train_data_for_Bert(data, text_encoder, text_head):
    caps, masks = data
    words_emb, sent_emb = encode_Bert_tokens(text_encoder, text_head, caps, masks)
    return words_emb, sent_emb



def prepare_test_data(data, text_encoder):
    img1, img2, cap1, cap2, cap_len1, cap_len2, pair_label = data
    cap1, sorted_cap_len1, sorted_cap_idxs = sort_sents(cap1, cap_len1)
    words_emb1, sent_emb1 = encode_tokens(text_encoder, cap1, sorted_cap_len1)
    sent_emb1 = rm_sort(sent_emb1, sorted_cap_idxs)
    words_emb1 = rm_sort(words_emb1, sorted_cap_idxs)

    cap2, sorted_cap_len2, sorted_cap_idxs = sort_sents(cap2, cap_len2)
    words_emb2, sent_emb2 = encode_tokens(text_encoder, cap2, sorted_cap_len2)
    sent_emb2 = rm_sort(sent_emb2, sorted_cap_idxs)
    words_emb2 = rm_sort(words_emb2, sorted_cap_idxs)

    return img1, img2, words_emb1, words_emb2, sent_emb1, sent_emb2, pair_label 



def prepare_test_data_Bert(data, text_encoder, text_head):
    img1, img2, caption1, caption2, mask1, mask2, pair_label = data

    words_emb1, sent_emb1 = encode_Bert_tokens(text_encoder, text_head, caption1, mask1)
    words_emb2, sent_emb2 = encode_Bert_tokens(text_encoder, text_head, caption2, mask2)

    return (img1, img2, 
            words_emb1, words_emb2, 
            sent_emb1, sent_emb2, 
            pair_label) 



############## dataloader #############
def prepare_dataloader(args, split, transform):
    if transform is not None:
        image_transform = transform
    else:
        image_transform = None 

    if args.en_type == "BERT":
        train_filenames, train_captions, train_att_masks, \
        valid_filenames, valid_captions, valid_att_masks, \
        test_filenames, test_captions, test_att_masks =  load_text_data_Bert(args.data_dir, args)

        if (split == "train"):
            train_ds = TrainDataset(train_filenames, train_captions, train_att_masks, 
                                    transform=image_transform, split="train", args=args)


        elif (split == "valid"):
            valid_ds = TestDataset(valid_filenames, valid_captions, valid_att_masks, 
                                transform=image_transform, split="valid", args=args)

        elif (split == "test"):
            test_ds =  TestDataset(test_filenames, test_captions, test_att_masks, 
                                transform=image_transform, split="test", args=args)


    elif args.en_type == "LSTM":
        train_names, train_captions, valid_names, valid_captions, \
        test_names, test_captions, ixtoword, wordtoix, n_words = \
            load_text_data(args.data_dir, args.captions_per_image)

        if (split == "train"):
            train_ds = TrainDataset(train_names, train_captions, None, ixtoword, wordtoix, n_words,
                                        transform=image_transform, split="train", args=args)

        elif (split == "valid"):
            valid_ds =  TestDataset(valid_names, valid_captions, None, ixtoword, wordtoix, n_words,
                                        transform=image_transform, split="valid", args=args)
        elif (split == "test"):
            test_ds =  TestDataset(test_names, test_captions, None, ixtoword, wordtoix, n_words,
                                        transform=image_transform, split="test", args=args)


    if (split == "train"):
        train_dl = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=args.batch_size, 
            drop_last=True,
            num_workers=args.num_workers, 
            shuffle=True)
        
        return train_dl, train_ds

    elif (split == "valid"):
        valid_dl = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=args.batch_size, 
            drop_last=False,
            num_workers=args.num_workers, 
            shuffle=False)

        return valid_dl, valid_ds 


    elif (split == "test"):
        test_dl = torch.utils.data.DataLoader(
            test_ds, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            drop_last = False, 
            shuffle=False)

        return test_dl, test_ds