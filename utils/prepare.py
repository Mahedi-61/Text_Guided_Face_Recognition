import os.path as osp

import torch
import torchvision.transforms as transforms
from utils.train_dataset import TextImgTrainDataset
from utils.test_dataset import TextImgTestDataset 

from utils.utils import load_model_weights, load_only_model_for_image_rec
from models.DAMSM import RNN_ENCODER, CNN_ENCODER
from models.GAN import NetG
from models.resnet import resnet_face18 
from utils.dataset_utils import load_text_data, load_text_data_Bert

###########   preparation   ############
def prepare_image_encoder(args):
    device = args.device
    
    # image encoder
    image_encoder = CNN_ENCODER()
    img_encoder_path = args.TEXT.DAMSM_NAME.replace('text_encoder', 'image_encoder')
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
    text_encoder = RNN_ENCODER(args, nhidden=args.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(args.TEXT.DAMSM_NAME, map_location='cpu')
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
    netG = NetG(num_classes = args.num_classes)
    netG = torch.nn.DataParallel(netG, device_ids=args.gpu_id).to(device)
    return model, netG


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
                            load_text_data(args.data_dir, args.embeddings_num)

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
