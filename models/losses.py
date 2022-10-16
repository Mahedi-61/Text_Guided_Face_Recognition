import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.attention import func_attention


# ################## Loss for matching text-image ###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    #Returns cosine similarity between x1 and x2, computed along dim.
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels, class_ids, batch_size, args, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.bool)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.BoolTensor(masks)
        if args.CUDA:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * args.TRAIN.SMOOTH.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)

    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None

    return loss0, loss1



def words_loss(img_features, words_emb, labels, cap_lens, class_ids, batch_size, args):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []

    if args.using_BERT == False:
        cap_lens = cap_lens.data.tolist()

    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.bool)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))

        # Get the i-th text description
        if args.using_BERT == False: words_num = cap_lens[i]
        elif args.using_BERT: words_num = args.bert_words_num - 1 #removing [cls] token 

        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()

        word = word.repeat(batch_size, 1, 1)
        context = img_features

        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """

        weiContext, attn = func_attention(word, context, args.TRAIN.SMOOTH.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()

        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)

        row_sim.mul_(args.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        masks = torch.BoolTensor(masks)
        if args.CUDA:
            masks = masks.cuda()

    similarities = similarities * args.TRAIN.SMOOTH.GAMMA3

    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)

    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps




def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD



def word_level_correlation(img_features, words_emb, cap_lens, batch_size, class_ids, labels, args):

    masks = []
    att_maps = []
    result = 0
    cap_lens = cap_lens.data.tolist()
    similar_list = []
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))

        words_num = cap_lens[i]

        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        context = img_features[i, :, :, :].unsqueeze(0).contiguous()
       
        weiContext, attn = func_attention(word, context, args.TRAIN.SMOOTH.GAMMA1)
       
        aver = torch.mean(word,2)
        averT = aver.unsqueeze(1)
        res_word = torch.bmm(averT, word)
        res_softmax = F.softmax(res_word, 2)
        res_softmax = res_softmax.repeat(1, weiContext.size(1), 1)
        self_weiContext = weiContext * res_softmax

        word = word.transpose(1, 2).contiguous()
        self_weiContext = self_weiContext.transpose(1, 2).contiguous()
        word = word.view(words_num, -1)
        self_weiContext = self_weiContext.view(words_num, -1)
        
        row_sim = cosine_similarity(word, self_weiContext)
        row_sim = row_sim.view(1, words_num)

        row_sim.mul_(args.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)
        similar_list.append(F.sigmoid(row_sim[0,0]))

    similar_list = torch.tensor(similar_list, requires_grad=False).cuda()
    result = nn.BCELoss()(similar_list, labels)

    return result