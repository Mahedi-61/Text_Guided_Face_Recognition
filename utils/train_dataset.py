import torch.utils.data as data
import os
import numpy as np
import numpy.random as random
from utils.dataset_utils import *

################################################################
#                    Train Dataset
################################################################

class TrainDataset(data.Dataset):
    def __init__(self, filenames, captions, att_masks, ixtoword=None, wordtoix=None, 
                    n_words=None, transform=None, split="train", args=None):

        print("\n############## Loading %s dataset ################" % split)
        self.transform = transform
        self.embeddings_num = args.captions_per_image
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        self.en_type = args.en_type
        self.model_type = args.model_type
        self.split = split 

        if args.en_type == "BERT": 
            self.filenames = filenames
            self.captions = captions 
            self.att_masks = att_masks
            self.word_num = args.bert_words_num 

        elif args.en_type == "LSTM":
            self.filenames = filenames
            self.captions = captions 
            self.ixtoword = ixtoword
            self.wordtoix = wordtoix
            self.n_words = n_words
            self.word_num = args.lstm_words_num 

        split_dir = os.path.join(self.data_dir, self.split)
        self.class_id = load_class_id(split_dir)


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
        data_dir = os.path.join(self.data_dir, "images")

        img_extension = ".jpg" # works for all dataset 

        img_name = os.path.join(data_dir, self.split, key + img_extension)
        imgs = get_imgs(img_name, self.split, self.transform, self.model_type)

        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix

        if self.en_type == "BERT": 
            caps, mask = self.captions[sent_ix], self.att_masks[sent_ix]
            return imgs, caps, mask, key, cls_id 

        elif self.en_type == "LSTM": 
            caps, cap_len = self.get_caption(new_sent_ix)
            return imgs, caps, cap_len, key, cls_id 


    def __len__(self):
        return len(self.filenames)