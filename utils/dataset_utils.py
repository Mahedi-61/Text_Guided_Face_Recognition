from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from PIL import Image
import pickle


def sort_sents(captions, caption_lens):
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(caption_lens, 0, True)
    captions = captions[sorted_cap_indices].squeeze()
    captions = Variable(captions).cuda()
    sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    return captions, sorted_cap_lens, sorted_cap_indices


def encode_tokens(text_encoder, caption, cap_lens):
    with torch.no_grad():
        if hasattr(text_encoder, 'module'):
            hidden = text_encoder.module.init_hidden(caption.size(0))
        else:
            hidden = text_encoder.init_hidden(caption.size(0))
        words_embs, sent_emb = text_encoder(caption, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    return sent_emb, words_embs 


def rm_sort(caption, sorted_cap_idxs):
    non_sort_cap = torch.empty_like(caption)
    for idx, sort in enumerate(sorted_cap_idxs):
        non_sort_cap[sort] = caption[idx]
    return non_sort_cap


def get_imgs(img_path, config, bbox=None, transform=None):

    if config == "DAMSM":
        img = Image.open(img_path).convert('RGB')
        norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    elif config == "FE":
        img = Image.open(img_path).convert('L')
        norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])

    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    img = norm(img)
    return img


def load_bbox(data_dir, split):
    bbox_path = os.path.join(data_dir, "CUB_200_2011/bounding_boxes.txt")
    df_bounding_boxes = pd.read_csv(bbox_path,
                                    delim_whitespace=True,
                                    header=None).astype(int)
    
    filepath = os.path.join(data_dir, "CUB_200_2011/" + split + "_images.txt")
    df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
    filenames = df_filenames[1].tolist()
    print("Total Images: ", len(filenames))

    filename_bbox = {img_file[:-4]: [] for img_file in filenames}
    numImgs = len(filenames)
    for i in range(0, numImgs):
        # bbox = [x-left, y-top, width, height]
        bbox = df_bounding_boxes.iloc[i][1:].tolist()
        key = filenames[i][:-4]
        filename_bbox[key] = bbox
    return filename_bbox


def load_captions(data_dir, filenames, embeddings_num):
    all_captions = []
    for i in range(len(filenames)):
        cap_path = os.path.join(data_dir, "text", filenames[i] + ".txt")
    
        with open(cap_path, "r") as f:
            captions = f.read().encode('utf-8').decode('utf8').split('\n')
            cnt = 0
            for cap in captions:
                if len(cap) == 0: continue
                cap = cap.replace("\ufffd\ufffd", " ")

                # picks out sequences of alphanumeric characters as tokens and drops everything else
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(cap.lower())
                
                if len(tokens) == 0:
                    print('cap', cap)
                    continue
                tokens_new = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0:
                        tokens_new.append(t)
                
                all_captions.append(tokens_new)
                cnt += 1
                if cnt == embeddings_num:
                    break

            if cnt < embeddings_num:
                print('ERROR: the captions for %s less than %d' % (filenames[i], cnt))
    return all_captions


def load_text_data(data_dir, split, embeddings_num):
    filepath = os.path.join(data_dir, 'captions_DAMSM.pickle')

    if not os.path.isfile(filepath):
        train_names = load_filenames(data_dir, 'train')
        test_names = load_filenames(data_dir, 'test')

        train_captions = load_captions(data_dir, train_names, embeddings_num)
        test_captions = load_captions(data_dir, test_names, embeddings_num)

        train_captions, test_captions, ixtoword, wordtoix, n_words = \
            build_dictionary(train_captions, test_captions)

        with open(filepath, 'wb') as f:
            pickle.dump([train_captions, test_captions, ixtoword, wordtoix], 
                            f, protocol=2)
            print('\nSave to: ', filepath)
    else:
        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            train_captions, test_captions = x[0], x[1]
            ixtoword, wordtoix = x[2], x[3]
            del x
            n_words = len(ixtoword)
            #print('\nCaptions load from: ', filepath)

    if split == 'train':
        train_names = load_filenames(data_dir, 'train')
        captions = train_captions
        filenames = train_names

    elif split=='test':
        test_names = load_filenames(data_dir, 'test')
        captions = test_captions
        filenames = test_names

    return filenames, captions, ixtoword, wordtoix, n_words


def build_dictionary(train_captions, test_captions):
    word_counts = defaultdict(float)
    captions = train_captions + test_captions
    for sent in captions:
        for word in sent:
            word_counts[word] += 1

    vocab = [w for w in word_counts if word_counts[w] >= 0]
    ixtoword = {}
    ixtoword[0] = '<end>'
    wordtoix = {}
    wordtoix['<end>'] = 0
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    train_captions_new = []
    for t in train_captions:
        rev = []
        for w in t:
            if w in wordtoix:
                rev.append(wordtoix[w])
        # rev.append(0)  # do not need '<end>' token
        train_captions_new.append(rev)

    test_captions_new = []
    for t in test_captions:
        rev = []
        for w in t:
            if w in wordtoix:
                rev.append(wordtoix[w])
        # rev.append(0)  # do not need '<end>' token
        test_captions_new.append(rev)

    return [train_captions_new, test_captions_new, ixtoword, wordtoix, len(ixtoword)]


def load_filenames(data_dir, split):
    filepath = os.path.join(data_dir, split, "filenames.pickle")

    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('\nLoad %s filenames from: %s (%d)' % (split, filepath, len(filenames)))
        print("Sample %s filenames: %s" % (split, filenames[0]))
    else:
        filenames = []
    return filenames


def load_class_id(data_dir):
    filepath = os.path.join(data_dir, "class_info.pickle")

    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            class_id = pickle.load(f, encoding="bytes")

    print('\nLoad %s class_info from: %s (%d)' % (data_dir, filepath, len(class_id)))
    return class_id