import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np 
import os
import numpy.random as random
from utils.dataset_utils import * 


################################################################
#                    Test Dataset
################################################################
class TestDataset(data.Dataset):
    def __init__(self, filenames, captions, att_masks, ixtoword=None, wordtoix=None, 
                    n_words=None, transform=None, split="", args=None):
        
        print("\n############## Loading %s dataset ################" % split)
        self.split= split
        self.transform = transform
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        self.embeddings_num = args.captions_per_image
        self.model_type = args.model_type 
        #self.is_ident = args.is_ident 
        self.filenames = filenames
        self.captions = captions 

        self.en_type = args.en_type 
        if args.en_type == "BERT": 
            self.word_num = args.bert_words_num
            self.att_masks = att_masks

        elif args.en_type == "LSTM":
            self.word_num = args.lstm_words_num
            self.ixtoword = ixtoword
            self.wordtoix = wordtoix
            self.n_words = n_words

        self.class_id = load_class_id(os.path.join(self.data_dir, self.split))

        if split == "test":
            self.test_pair_list = args.test_pair_list
        elif split == "valid":
             self.test_pair_list = args.valid_pair_list

        self.imgs_pair, self.pair_label = self.get_test_list()


    def get_test_list(self):
        with open(self.test_pair_list, 'r') as fd:
            pairs = fd.readlines()
        imgs_pair = []
        pair_label = []
        for pair in pairs:
            splits = pair.split(" ")
            imgs = [splits[0], splits[1]]
            imgs_pair.append(imgs)
            pair_label.append(int(splits[2]))
        return imgs_pair, pair_label


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
        imgs = self.imgs_pair[index]
        pair_label = self.pair_label[index]
        data_dir = os.path.join(self.data_dir, "images")

        img1_name = os.path.join(imgs[0].split("_")[0], imgs[0])
        img2_name = os.path.join(imgs[1].split("_")[0], imgs[1])

        img1_path = os.path.join(data_dir, self.split, img1_name)
        img2_path = os.path.join(data_dir, self.split, img2_name)

        key1 = img1_name[:-4]
        key2 = img2_name[:-4]

        img1 = get_imgs(img1_path, self.split, self.transform, self.model_type)
        img2 = get_imgs(img2_path, self.split, self.transform, self.model_type)

        real_index1 = self.filenames.index(key1)
        real_index2 = self.filenames.index(key2)

        #randomly select a sentence
        #sent_ix1 = random.randint(0, self.embeddings_num)
        #select the first sentence 
        sent_ix1 = 0
        new_sent_ix1 = real_index1 * self.embeddings_num + sent_ix1
        
        # randomly select another sentence
        sent_ix2 = 0 #random.randint(0, self.embeddings_num)
        new_sent_ix2 = real_index2 * self.embeddings_num + sent_ix2
        
        if self.en_type == "BERT": 
            cap1, mask1 = self.captions[new_sent_ix1], self.att_masks[new_sent_ix1]
            cap2, mask2 = self.captions[new_sent_ix2], self.att_masks[new_sent_ix2]
            return img1, img2, cap1, cap2, mask1, mask2, pair_label

        elif self.en_type == "LSTM":
            cap1, cap_len1 = self.get_caption(new_sent_ix1) 
            cap2, cap_len2 = self.get_caption(new_sent_ix2)
            return img1, img2, cap1, cap2, cap_len1, cap_len2, pair_label

        if (index % 50000 == 0): print(index) 

    def __len__(self):
        return len (self.imgs_pair)