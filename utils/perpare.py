import os, sys
import os.path as osp
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils


from utils.datasets import TextImgDataset as Dataset
from utils.utils import load_model_weights
from models.DAMSM import RNN_ENCODER, CNN_ENCODER
from models.GAN import NetG
from models.resnet import resnet_face18 

###########   preparation   ############
def prepare_models(args):
    device = args.device
    n_words = args.vocab_size
    
    # image encoder
    image_encoder = CNN_ENCODER(args.TEXT.EMBEDDING_DIM)
    img_encoder_path = args.TEXT.DAMSM_NAME.replace('text_encoder', 'image_encoder')
    state_dict = torch.load(img_encoder_path, map_location='cpu')
    image_encoder = load_model_weights(image_encoder, state_dict)
    # image_encoder.load_state_dict(state_dict)
    image_encoder.to(device)
    for p in image_encoder.parameters():
        p.requires_grad = False
    image_encoder.eval()

    # text encoder
    print("loading text encoder")
    text_encoder = RNN_ENCODER(n_words, nhidden=args.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(args.TEXT.DAMSM_NAME, map_location='cpu')
    text_encoder = load_model_weights(text_encoder, state_dict)
    text_encoder.cuda()
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()


    #archface model for image
    model = resnet_face18(use_se=args.use_se)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
    model.load_state_dict(torch.load(args.load_model_path))
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # GAN models
    netG = NetG(num_classes = args.num_classes)
    netG = torch.nn.DataParallel(netG, device_ids=args.gpu_id).to(device)
                                                          
    return image_encoder, text_encoder, model, netG


def prepare_dataset(args, split, transform):
    imsize = args.imsize
    if transform is not None:
        image_transform = transform

    elif args.CONFIG_NAME.find('celeba') != -1:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
    else:
        if (split == "train"):
            image_transform = transforms.Compose([
                transforms.Resize(144),
                transforms.RandomCrop(imsize),
                transforms.RandomHorizontalFlip()])

        elif (split == "test"):
            image_transform = transforms.Compose([
                transforms.Resize(144),
                transforms.RandomCrop(imsize)])

    return Dataset(split=split, transform=image_transform, args=args)


def get_train_dataloader(args, transform=None):
    train_dataset = prepare_dataset(args, split='train', transform=transform)
    

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        drop_last=True,
        num_workers=args.num_workers, 
        shuffle=True)
    return train_dataloader, train_dataset



def get_test_dataloader(args, transform=None):
    test_dataset = prepare_dataset(args, split='test', transform=transform)
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=False)

    return test_dataloader, test_dataset

