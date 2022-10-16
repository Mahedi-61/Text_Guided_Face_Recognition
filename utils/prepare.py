import os.path as osp

import torch
import torchvision.transforms as transforms
from utils.train_dataset import TextImgTrainDataset
from utils.test_dataset import TextImgTestDataset 

from utils.utils import load_model_weights, load_only_model_for_image_rec
from models.models import RNN_ENCODER, ResNetFace_ENCODER, BERT_ENCODER, resnet_face18
from models.fusion_nets import LinearFusion, CrossAttention 
from utils.dataset_utils import *


###########   model   ############
def prepare_image_encoder(args):
    device = args.device
    
    # image encoder
    image_encoder = ResNetFace_ENCODER()
    img_encoder_path = args.damsm_encoder_path.replace('text_encoder', 'image_encoder')
    state_dict = torch.load(img_encoder_path, map_location='cpu')
    image_encoder = load_model_weights(image_encoder, state_dict)
    image_encoder.to(device)

    for p in image_encoder.parameters():
        p.requires_grad = False
    image_encoder.eval()
    return image_encoder



def prepare_text_encoder(args):
    device = args.device

    # text encoder
    print("loading text encoder")
    if args.using_BERT == True:
        text_encoder = BERT_ENCODER(args)
        text_encoder = torch.nn.DataParallel(text_encoder, device_ids=args.gpu_id).cuda()
        state_dict = torch.load(args.damsm_encoder_path)
        text_encoder.load_state_dict(state_dict['model'])

    elif args.using_BERT == False:
        text_encoder = RNN_ENCODER(args, nhidden=args.embedding_dim)
        state_dict = torch.load(args.damsm_encoder_path, map_location='cpu')
        text_encoder = load_model_weights(text_encoder, state_dict)
        text_encoder.cuda()

    for p in text_encoder.parameters():
        p.requires_grad = False

    text_encoder.eval()
    return text_encoder



def prepare_models(args):
    device = args.device

    #archface model for image
    model = resnet_face18(use_se=args.use_se)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
    model = load_only_model_for_image_rec(model, args.load_model_path, args.prev_weight)
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # GAN models

    if args.fusion_type == "linear":
        net = LinearFusion()
    elif args.fusion_type == "cross_attention":
        net = CrossAttention(channel_dim = 256)

    net = torch.nn.DataParallel(net, device_ids=args.gpu_id).to(device)
    return model, net



################ data ##############
def get_one_batch_train_data(dataloader, text_encoder):
    data = next(iter(dataloader))
    imgs, sent_emb, words_embs, keys, label = prepare_train_data(data, text_encoder)
    return imgs, words_embs, sent_emb


def prepare_train_data(data, text_encoder):
    imgs, captions, caption_lens, keys, label = data
    captions, sorted_cap_lens, sorted_cap_idxs = sort_sents(captions, caption_lens)
    sent_emb, words_embs = encode_tokens(text_encoder, captions, sorted_cap_lens)
    sent_emb = rm_sort(sent_emb, sorted_cap_idxs)
    words_embs = rm_sort(words_embs, sorted_cap_idxs)
    imgs = Variable(imgs).cuda()
    return imgs, sent_emb, words_embs, keys, label, caption_lens


def prepare_train_data_for_Bert(data, text_encoder):
    imgs, caps, masks, keys, cls_ids = data
    sent_emb, words_embs = encode_Bert_tokens(text_encoder, caps, masks)
    imgs = Variable(imgs).cuda()
    return imgs, sent_emb, words_embs, keys, cls_ids


def get_one_batch_train_data_Bert(dataloader):
    data = next(iter(dataloader))
    imgs, sent_emb, words_emb, keys, cls_ids = prepare_train_data_for_Bert(data)
    return imgs, words_emb, sent_emb


def prepare_test_data(data, text_encoder):
    img1, img2, cap1, cap2, cap_len1, cap_len2, pair_label = data

    cap1, sorted_cap_len1, sorted_cap_idxs = sort_sents(cap1, cap_len1)
    sent_emb1, words_emb1 = encode_tokens(text_encoder, cap1, sorted_cap_len1)
    sent_emb1 = rm_sort(sent_emb1, sorted_cap_idxs)
    words_emb1 = rm_sort(words_emb1, sorted_cap_idxs)

    cap2, sorted_cap_len2, sorted_cap_idxs = sort_sents(cap2, cap_len2)
    sent_emb2, words_emb2 = encode_tokens(text_encoder, cap2, sorted_cap_len2)
    sent_emb2 = rm_sort(sent_emb2, sorted_cap_idxs)
    words_emb2 = rm_sort(words_emb2, sorted_cap_idxs)

    return img1, img2, sent_emb1, sent_emb2, words_emb1, words_emb2, pair_label  


def prepare_test_data_Bert(data, text_encoder):
    img1, img2, caption1, caption2, mask1, mask2, pair_label = data

    sent_emb1, words_emb1 = encode_Bert_tokens(text_encoder, caption1, mask1)
    sent_emb2, words_emb2  = encode_Bert_tokens(text_encoder, caption2, mask2)

    return img1, img2, sent_emb1, sent_emb2, words_emb1, words_emb2, pair_label 



############## dataloader #############
def prepare_dataloader(args, split, transform):
    imsize = args.img_size
    if transform is not None:
        image_transform = transform

    else:
        if (split == "train") :
            image_transform = transforms.Compose([
                #transforms.Resize(144),
                transforms.RandomCrop(imsize),
                transforms.RandomHorizontalFlip()])

        elif (split == "test"):
            image_transform = transforms.Compose([
                #transforms.Resize(144),
                transforms.RandomCrop(imsize)])

    if args.using_BERT == True: 
        train_filenames, train_captions, train_att_masks, \
        test_filenames, test_captions, test_att_masks =  load_text_data_Bert(args.data_dir, args)
        if (split == "train"):
            train_ds = TextImgTrainDataset(train_filenames, train_captions, train_att_masks, 
                                    transform=image_transform, split="train", args=args)


            valid_ds = TextImgTrainDataset(test_filenames, test_captions, test_att_masks, 
                            transform=image_transform, split="valid", args=args)

        elif (split == "test"):
            test_ds =  TextImgTestDataset(test_filenames, test_captions, test_att_masks, 
                                        transform=image_transform, args=args)


    elif args.using_BERT == False:
        train_names, train_captions, test_names, test_captions, ixtoword, wordtoix, n_words = \
                            load_text_data(args.data_dir, args.captions_per_image)

        if (split == "train"):
            train_ds = TextImgTrainDataset(train_names, train_captions, None, ixtoword, wordtoix, n_words,
                            transform=image_transform, split="train", args=args)


            valid_ds = TextImgTrainDataset(test_names, test_captions, None, ixtoword, wordtoix, n_words,
                            transform=image_transform, split="valid", args=args)

        elif (split == "test"):
            test_ds =  TextImgTestDataset(test_names, test_captions, None, ixtoword, wordtoix, n_words,
                                        transform=image_transform, args=args)


    if (split == "train"):
        train_dl = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=args.train_batch_size, 
            drop_last=True,
            num_workers=args.num_workers, 
            shuffle=True)

        valid_dl = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=args.test_batch_size, 
            drop_last=True,
            num_workers=args.num_workers, 
            shuffle=True)

        return train_dl, train_ds, valid_dl, valid_ds 


    elif (split == "test"):
        test_dl = torch.utils.data.DataLoader(
            test_ds, 
            batch_size=args.test_batch_size, 
            num_workers=args.num_workers, 
            shuffle=False)

        return test_dl, test_ds
