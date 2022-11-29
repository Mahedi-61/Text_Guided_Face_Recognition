import torch.utils.data as data
import os
import numpy as np
import numpy.random as random
from utils.dataset_utils import *

################################################################
#                    Train Dataset
################################################################

class TextImgTrainDataset(data.Dataset):
    def __init__(self, filenames, captions, att_masks, ixtoword=None, wordtoix=None, 
                    n_words=None, transform=None, split="train", args=None):

        print("\n############## Loading %s dataset ################" % split)
        self.transform = transform
        self.embeddings_num = args.captions_per_image
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
            self.word_num = args.bert_words_num 

        elif args.using_BERT == False:
            self.filenames = filenames
            self.captions = captions 
            self.ixtoword = ixtoword
            self.wordtoix = wordtoix
            self.n_words = n_words
            self.word_num = args.lstm_words_num 

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

        elif self.config == "Fusion":
            data_dir = os.path.join(self.data_dir, "celeba")
            img_extension = ".png"
            img_folder = "train_images"

        if self.config == "Pretrain":
            data_dir = os.path.join(self.data_dir, "celeba")
            if self.split == "train": img_folder = "train_images_DAMSM"
            elif self.split == "test": img_folder = "test_images_DAMSM"
            img_extension = ".png"

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