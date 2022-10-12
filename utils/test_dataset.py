import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np 
import os
import numpy.random as random
from utils.dataset_utils import * 


def prepare_test_data(data, text_encoder):
    img1, img2, cap1, cap2, cap_len1, cap_len2, pair_label = data

    cap1, sorted_cap_len1, sorted_cap_idxs = sort_sents(cap1, cap_len1)
    sent_emb1, words_embs1 = encode_tokens(text_encoder, cap1, sorted_cap_len1)
    sent_emb1 = rm_sort(sent_emb1, sorted_cap_idxs)
    words_embs1 = rm_sort(words_embs1, sorted_cap_idxs)

    cap2, sorted_cap_len2, sorted_cap_idxs = sort_sents(cap2, cap_len2)
    sent_emb2, words_embs2 = encode_tokens(text_encoder, cap2, sorted_cap_len2)
    sent_emb2 = rm_sort(sent_emb2, sorted_cap_idxs)
    words_embs2 = rm_sort(words_embs2, sorted_cap_idxs)

    img1 = Variable(img1).cuda()
    img2 = Variable(img2).cuda()
    return img1, img2, sent_emb1, sent_emb2, pair_label 


################################################################
#                    Test Dataset
################################################################
class TextImgTestDataset(data.Dataset):
    def __init__(self, filenames, captions, att_masks, ixtoword=None, wordtoix=None, 
                    n_words=None, transform=None, args=None):
        self.split= "test"
        self.transform = transform
        self.word_num = args.TEXT.WORDS_NUM
        self.embeddings_num = args.TEXT.CAPTIONS_PER_IMAGE
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        
        if self.data_dir.find('birds') != -1:
            self.bbox = load_bbox(self.data_dir, self.split)
        else:
            self.bbox = None

        self.using_BERT = args.using_BERT 
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

        self.class_id = load_class_id(os.path.join(self.data_dir, self.split))
        self.test_pair_list = args.test_pair_list
        self.config = args.CONFIG_NAME
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

        if self.dataset_name == "birds":
            data_dir = os.path.join(self.data_dir, "CUB_200_2011")
        elif self.dataset_name == "celeba":
            data_dir = os.path.join(self.data_dir, "celeba")

        img1_name = '%s/test_images/%s' % (data_dir, imgs[0])
        img2_name = '%s/test_images/%s' % (data_dir, imgs[1])

        key1 = imgs[0][:-4]
        key2 = imgs[1][:-4]

        if self.dataset_name == "birds":
            bbox1 = self.bbox[key1]
            bbox2 = self.bbox[key2]
        else:
            bbox1 = None
            bbox2 = None 

        img1 = get_imgs(img1_name, self.config, bbox1, self.transform)
        img2 = get_imgs(img2_name, self.config, bbox2, self.transform)

        real_index1 = self.filenames.index(key1)
        real_index2 = self.filenames.index(key2)

        # randomly select a sentence
        sent_ix1 = random.randint(0, self.embeddings_num)
        new_sent_ix1 = real_index1 * self.embeddings_num + sent_ix1
        
        # randomly select another sentence
        sent_ix2 = random.randint(0, self.embeddings_num)
        new_sent_ix2 = real_index2 * self.embeddings_num + sent_ix2
        

        if self.using_BERT == True: 
            cap1, mask1 = self.captions[new_sent_ix1], self.att_masks[new_sent_ix1]
            cap2, mask2 = self.captions[new_sent_ix2], self.att_masks[new_sent_ix2]
            return img1, img2, cap1, cap2, mask1, mask2, pair_label

        elif self.using_BERT == False:
            cap1, cap_len1 = self.get_caption(new_sent_ix1) 
            cap2, cap_len2 = self.get_caption(new_sent_ix2)
            return img1, img2, cap1, cap2, cap_len1, cap_len2, pair_label


    def __len__(self):
        return len (self.imgs_pair)