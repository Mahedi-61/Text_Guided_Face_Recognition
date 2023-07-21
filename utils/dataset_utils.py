from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from PIL import Image
import _pickle as pickle
import gc
from transformers import BertTokenizer
#from transformers import AutoTokenizer


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
        #words_embs, sent_emb = text_head(words_embs, sent_emb)
    return words_embs.detach(), sent_emb.detach()



def encode_Bert_tokens(text_encoder, text_head, caption, mask):
    caption = Variable(caption).cuda()
    mask = Variable(mask).cuda()

    with torch.no_grad():
         words_emb, sent_emb_org = text_encoder(caption, mask)
         words_emb, word_vector, sent_emb = text_head(words_emb, sent_emb_org)

    return words_emb.detach(), word_vector.detach(), sent_emb.detach()


def rm_sort(caption, sorted_cap_idxs):
    non_sort_cap = torch.empty_like(caption)
    for idx, sort in enumerate(sorted_cap_idxs):
        non_sort_cap[sort] = caption[idx]
    return non_sort_cap



def get_imgs(img_path, split, transform=None, model_type="arcface"):

    img = Image.open(img_path).convert('RGB')

    if transform == None:
        if split == "train":
            transform = transforms.Compose([
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip (p = 0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        elif split == "test" or split == "valid":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img = transform(img)
    if model_type == "adaface": 
        permute = [2, 1, 0]
        img = img[permute, :, :] #RGB --> BGR

    return img



def load_captions_Bert(data_dir, filenames, args):
    # convert the raw text into a list of tokens.
    # attention_mask (which tokens should be used by the model 1 - use or 0 - don’t use).
    tokenizer = BertTokenizer.from_pretrained(args.bert_config)
    all_captions = []
    all_attention_mask = []

    for i in range(len(filenames)):
        cap_path = os.path.join(data_dir, "text", filenames[i] + ".txt")
    
        with open(cap_path, "r") as f:
            captions = f.read().encode('utf-8').decode('utf8').split('\n')
            cnt = 0
            for cap in captions:
                if len(cap) == 0: continue
                cap = cap.replace("\ufffd\ufffd", " ")
                
                encoding = tokenizer.encode_plus(
                            cap,
                            add_special_tokens=True,
                            max_length = args.bert_words_num,
                            return_token_type_ids=False,
                            padding='max_length',
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors='pt')

                input_ids=encoding["input_ids"].flatten()
                attention_mask=encoding["attention_mask"].flatten()

                all_captions.append(input_ids)
                all_attention_mask.append(attention_mask)

                cnt += 1
                if cnt == args.captions_per_image:
                    break

            if cnt < args.captions_per_image:
                print('ERROR: the captions for %s less than %d' % (filenames[i], cnt))

    del captions 
    return all_captions, all_attention_mask



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

    del captions 
    return all_captions



def load_text_data_Bert(data_dir, args):
    filepath = os.path.join(data_dir, 'captions_BERT.pickle')

    if not os.path.isfile(filepath):
        train_names = load_filenames(data_dir, 'train')
        valid_names = load_filenames(data_dir, 'valid')
        test_names = load_filenames(data_dir, 'test')

        train_captions, train_att_masks = load_captions_Bert(data_dir, train_names, args)
        valid_captions, valid_att_masks = load_captions_Bert(data_dir, valid_names, args)
        test_captions, test_att_masks = load_captions_Bert(data_dir, test_names, args)

        with open(filepath, 'wb') as f:
            pickle.dump([train_captions, train_att_masks, valid_captions, 
                        valid_att_masks, test_captions, test_att_masks], 
                        f, protocol=2)
            print('\nSave to: ', filepath)
    else:
        print("Loading captions_BERT.pickle")
        with open(filepath, 'rb') as f:
            gc.disable()
            x = pickle.load(f)
            gc.enable()
            train_captions, train_att_masks, valid_captions, valid_att_masks, \
                test_captions, test_att_masks = x[0], x[1], x[2], x[3], x[4], x[5]
        del x

    train_names = load_filenames(data_dir, 'train')
    valid_names = load_filenames(data_dir, 'valid')
    test_names = load_filenames(data_dir, 'test')

    print("loading complete")
    return (train_names, train_captions, train_att_masks, 
            valid_names, valid_captions, valid_att_masks, 
            test_names, test_captions, test_att_masks) 



def load_text_data(data_dir, embeddings_num):
    filepath = os.path.join(data_dir, 'captions_RNN.pickle')

    if not os.path.isfile(filepath):
        train_names = load_filenames(data_dir, 'train')
        valid_names = load_filenames(data_dir, 'valid')
        test_names = load_filenames(data_dir, 'test')

        train_captions = load_captions(data_dir, train_names, embeddings_num)
        valid_captions = load_captions(data_dir, valid_names, embeddings_num)
        test_captions = load_captions(data_dir, test_names, embeddings_num)

        train_captions, valid_captions, test_captions, ixtoword, wordtoix, n_words = \
            build_dictionary(train_captions, valid_captions, test_captions)

        with open(filepath, 'wb') as f:
            pickle.dump([train_captions, valid_captions, test_captions, ixtoword, wordtoix], 
                            f, protocol=2)
            print('\nSave to: ', filepath)
    else:
        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            train_captions, valid_captions, test_captions = x[0], x[1], x[2]
            ixtoword, wordtoix = x[3], x[4]
            del x
            n_words = len(ixtoword)

    train_names = load_filenames(data_dir, 'train')
    valid_names = load_filenames(data_dir, 'valid')
    test_names = load_filenames(data_dir, 'test')
    print("loading data complete")

    return (train_names, train_captions, valid_names, valid_captions, 
            test_names, test_captions, ixtoword, wordtoix, n_words)



def build_dictionary(train_captions, valid_captions, test_captions):
    word_counts = defaultdict(float)
    captions = train_captions + valid_captions + test_captions 
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

    valid_captions_new = []
    for t in valid_captions:
        rev = []
        for w in t:
            if w in wordtoix:
                rev.append(wordtoix[w])
        # rev.append(0)  # do not need '<end>' token
        valid_captions_new.append(rev)

    test_captions_new = []
    for t in test_captions:
        rev = []
        for w in t:
            if w in wordtoix:
                rev.append(wordtoix[w])
        # rev.append(0)  # do not need '<end>' token
        test_captions_new.append(rev)

    return [train_captions_new, valid_captions_new, test_captions_new, ixtoword, wordtoix, len(ixtoword)]


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
            gc.disable()
            class_id = pickle.load(f, encoding="bytes")
            gc.enable()

    print('Load class_info from: %s (%d)' % (filepath, len(class_id)))
    return class_id