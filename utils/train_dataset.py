import torch
import torch.utils.data as data
from torch.autograd import Variable
from utils.dataset_utils import *

import os
import numpy as np
import numpy.random as random



def get_one_batch_data(dataloader, text_encoder, args):
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
    return imgs, sent_emb, words_embs, keys, label 


def prepare_train_data_for_DAMSM(data, text_encoder):
    imgs, captions, caption_lens, keys, cls_ids = data
    captions, sorted_cap_lens, sorted_cap_idxs = sort_sents(captions, caption_lens)
    sent_emb, words_embs = encode_tokens(text_encoder, captions, sorted_cap_lens)
    sent_emb = rm_sort(sent_emb, sorted_cap_idxs)
    words_embs = rm_sort(words_embs, sorted_cap_idxs)
    imgs = Variable(imgs).cuda()
    return imgs, sent_emb, words_embs, keys, cls_ids, caption_lens



####################### for BERT ################################
def prepare_train_data_for_Bert(data, text_encoder):
    imgs, caps, masks, keys, cls_ids = data
    sent_emb, words_embs = encode_Bert_tokens(text_encoder, caps, masks)
    imgs = Variable(imgs).cuda()
    return imgs, sent_emb, words_embs, keys, cls_ids


def get_one_batch_data_Bert(dataloader):
    data = next(iter(dataloader))
    imgs, sent_emb, words_emb, keys, cls_ids = prepare_train_data_for_Bert(data)
    return imgs, words_emb, sent_emb


class TextImgTrainDataset(data.Dataset):
    def __init__(self, filenames, captions, att_masks, ixtoword=None, wordtoix=None, 
                    n_words=None, transform=None, split="train", args=None):

        print("############## Loading %s dataset ################" % split)
        self.transform = transform
        self.word_num = args.TEXT.WORDS_NUM
        self.embeddings_num = args.TEXT.CAPTIONS_PER_IMAGE
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        self.using_BERT = args.using_BERT
        
        if split == "train":
            self.split = split 

        elif split == "valid":
            self.split = "test"

        if self.data_dir.find('birds') != -1:
            self.bbox = load_bbox(self.data_dir, self.split)
        else:
            self.bbox = None

        if args.using_BERT == True: 
            self.filenames = filenames
            self.captions = captions 
            self.att_masks = att_masks

        elif args.using_BERT == False:
            self.filenames = filenames
            self.captions = captions 
            self.ixtoword = ixtoword
            self.wordtoix = wordtoix
            self.n_words = n_words

        split_dir = os.path.join(self.data_dir, self.split)
        self.class_id = load_class_id(split_dir)
        self.config = args.CONFIG_NAME


    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')

        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)

        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.word_num, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.word_num:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.word_num]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.word_num

        return x, x_len


    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]
        data_dir = self.data_dir
        
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None

        if self.dataset_name == "birds":
            data_dir = os.path.join(self.data_dir, "CUB_200_2011")
            img_extension = ".jpg"

        elif self.dataset_name == "celeba":
            data_dir = os.path.join(self.data_dir, "celeba")
            img_extension = ".png"

        if self.config == "DAMSM":
            if self.split == "train": img_folder = "train_images_DAMSM"
            elif self.split == "test": img_folder = "test_images_DAMSM"

        else:
            img_folder = "train_images"

        img_name = "%s/%s/%s%s" % (data_dir, img_folder, key, img_extension)
        imgs = get_imgs(img_name, self.config, bbox, self.transform)

        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix

        if self.using_BERT == True: 
            caps, mask = self.captions[sent_ix], self.att_masks[sent_ix]
            return imgs, caps, mask, key, cls_id 

        elif self.using_BERT == False: 
            caps, cap_len = self.get_caption(new_sent_ix)
            return imgs, caps, cap_len, key, cls_id 


    def __len__(self):
        return len(self.filenames)