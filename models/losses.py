import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.attention import func_attention
from torch.nn.parameter import Parameter
from einops import rearrange
from torch.autograd import Variable


# ################## Loss for matching text-image ###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels, class_ids, batch_size, args, eps=1e-8):
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(bool)
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
        img_features(context): batch x nef x 14 x 14
    """
    masks = []
    att_maps = []
    similarities = []

    if not args.en_type == "BERT":
        cap_lens = cap_lens.data.tolist()

    for i in range(batch_size):
        """
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(bool)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        """

        # Get the i-th text description
        if args.en_type == "LSTM": words_num = cap_lens[i]
        elif args.en_type == "BERT": words_num = args.bert_words_num - 2 #removing [cls] token 

        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        word = word.repeat(batch_size, 1, 1)
        context = img_features

        """
            word(query): batch x nef x words_num
            context: batch x nef x 14 x 14
            weiContext: batch x nef x words_num
            attn: batch x words_num x 14 x 14
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
    """
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        masks = torch.BoolTensor(masks)
        if args.CUDA:
            masks = masks.cuda()
    """
    similarities = similarities * args.TRAIN.SMOOTH.GAMMA3

    """
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    """
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


def clip_loss(text_embeddings, image_embeddings, args):
    logits = (text_embeddings @ image_embeddings.T) / args.temperature
    images_similarity = image_embeddings @ image_embeddings.T
    texts_similarity = text_embeddings @ text_embeddings.T

    targets = F.softmax(
        (images_similarity + texts_similarity) / 2 * args.temperature, dim=-1)

    texts_loss = cross_entropy(logits, targets, reduction='none')
    images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
    return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()



class CMPLoss(nn.Module):
    def __init__(self, is_CMPM, is_CMPC, num_classes, feature_dim):
        super(CMPLoss, self).__init__()
        self.CMPM = is_CMPM
        self.CMPC = is_CMPC
        self.epsilon = 1e-8
        self.num_classes = num_classes

        self.W = Parameter(torch.randn(feature_dim, num_classes))
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.W.data, gain=1)
        

    def compute_cmpc_loss(self, text_embeddings, image_embeddings, labels):
        """
        Cross-Modal Projection Classfication loss(CMPC)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
        """
        criterion = nn.CrossEntropyLoss(reduction='mean')
        self.W_norm = self.W / self.W.norm(dim=0)
        #labels_onehot = one_hot_coding(labels, self.num_classes).float()
        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        image_proj_text = torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
        text_proj_image = torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm

        image_logits = torch.matmul(image_proj_text, self.W_norm)
        text_logits = torch.matmul(text_proj_image, self.W_norm)
        
        #labels_one_hot = one_hot_coding(labels, num_classes)
        '''
        ipt_loss = criterion(input=image_logits, target=labels)
        tpi_loss = criterion(input=text_logits, target=labels)
        cmpc_loss = ipt_loss + tpi_loss
        '''
        cmpc_loss = criterion(image_logits, labels) + criterion(text_logits, labels)
        return cmpc_loss


    def compute_cmpm_loss(self, text_embeddings, image_embeddings, labels):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)
        
        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)
         
        i2t_pred = F.softmax(image_proj_text, dim=1)
        #i2t_loss = i2t_pred * torch.log((i2t_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon))
        
        t2i_pred = F.softmax(text_proj_image, dim=1)
        #t2i_loss = t2i_pred * torch.log((t2i_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))       
        return cmpm_loss


    def forward(self, text_embeddings, image_embeddings, labels):
        cmpc_loss = 0.0
        cmpm_loss = 0.0
        
        if self.CMPC:
            cmpc_loss = self.compute_cmpc_loss(text_embeddings, image_embeddings, labels)

        if self.CMPM:
            cmpm_loss = self.compute_cmpm_loss(text_embeddings, image_embeddings, labels)

        
        loss = cmpc_loss + cmpm_loss
        return loss, cmpc_loss, cmpm_loss
    


class ClipLoss(nn.Module):
    def __init__(
            self,
            cache_labels=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)

            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, text_features, image_features, args, logit_scale=1):
        device = args.device #image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return total_loss 
    


class FocalLoss(nn.Module):
    
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()



def global_loss(cnn_code, rnn_code, eps=1e-8, temp3=10.0):

    batch_size = cnn_code.shape[0]
    labels = Variable(torch.LongTensor(range(batch_size))).to(cnn_code.device)

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * temp3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()

    scores1 = scores0.transpose(0, 1)
    loss0 = nn.CrossEntropyLoss()(scores0, labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)
    return loss0 + loss1



class WordRegionAlignment(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.local_temperature = 0.1
        self.num_heads = 1
        self.bidirectional = True
        self.use_local_atten = False
        self.emb_dim = 256 

        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            self.emb_dim, self.num_heads, batch_first=True)
        
        # sentence local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            self.emb_dim, self.num_heads, batch_first=True)


    def forward(self, img_emb, region_emb, caption_emb, word_emb, word_attn):
        

        bz = img_emb.size(0)
        #labels = torch.arange(bz).type_as(word_emb).long()
        #mask = torch.from_numpy(np.array(sents)[:, 1:] == "[PAD]").type_as(batch["imgs"]).bool()

        if self.use_local_atten:
            word_atten_output, _ = self.word_local_atten_layer(
                                    word_emb, region_emb, region_emb)
        else:
            atten_sim = torch.bmm(word_emb, region_emb.permute(0, 2, 1))
            word_num = word_emb.size(1)
            # atten_sim[mask.unsqueeze(1).repeat(1, word_num, 1)] = float("-inf")
            atten_scores = F.softmax(atten_sim / self.local_temperature, dim=-1)  # bz, 196, 111
            word_atten_output = torch.bmm(atten_scores, region_emb)

        word_atten_output = F.normalize(word_atten_output, dim=-1)
        word_sim = torch.bmm(word_emb, word_atten_output.permute(0, 2, 1)) / self.local_temperature
        
        with torch.no_grad():
            atten_weights = word_attn.detach()
            word_atten_weights = []

            for i in range(bz):
                atten_weight = atten_weights[i]
                nonzero = atten_weight.nonzero().squeeze()
                low = torch.quantile(atten_weight[nonzero], 0.1)
                high = torch.quantile(atten_weight[nonzero], 0.9)
                atten_weight[nonzero] = atten_weight[nonzero].clip(low, high)
                word_atten_weights.append(atten_weight.clone())
            word_atten_weights = torch.stack(word_atten_weights)
            # TODO: maybe clip the tensor of 10 percentile and 90 percentile

        word_atten_weights /= word_atten_weights.sum(dim=1, keepdims=True)

        word_sim = torch.bmm(word_emb, word_atten_output.permute(0, 2, 1)) / self.local_temperature
        word_num = word_sim.size(1)
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")
        targets = torch.arange(word_num).type_as(word_emb).long().repeat(bz)

        loss_word_1 = torch.sum(F.cross_entropy(
                                word_sim_1, targets, reduction="none") * 
                                word_atten_weights.view(-1)) / bz

        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = torch.sum(F.cross_entropy(
                                    word_sim_2, targets, reduction="none") * 
                                    word_atten_weights.view(-1)) / bz

        loss_word = (loss_word_1 + loss_word_2) / 2.0

        """
        if self.hparams.bidirectional:
            # Try not use atten layer
            if self.use_local_atten:
                patch_atten_output, _ = self.patch_local_atten_layer(
                    patch_emb_q, word_emb_q, word_emb_q, key_padding_mask=mask)
            else:
                atten_sim = torch.bmm(patch_emb_q, word_emb_q.permute(0, 2, 1))
                patch_num = patch_emb_q.size(1)
                atten_sim[mask.unsqueeze(1).repeat(1, patch_num, 1)] = float("-inf")
                
                atten_scores = F.softmax(atten_sim / self.local_temperature, dim=-1)  # bz, 196, 111
                patch_atten_output = torch.bmm(atten_scores, word_emb_q)

            # patch_atten_output: bz, 196, 128
            if "vit" not in self.hparams.img_encoder:
                patch_atten_output = F.normalize(patch_atten_output, dim=-1)
                patch_num = patch_atten_output.size(1)
                patch_atten_weights = torch.ones(
                    bz, patch_num).type_as(batch["imgs"]) / patch_num

            else:
                with torch.no_grad():
                    img_attn_map = self.img_encoder_q.model.blocks[-1].attn.attention_map.detach(
                    )
                    atten_weights = img_attn_map[:, :, 0, 1:].mean(dim=1)
                    patch_atten_weights = []

                    for i in range(bz):
                        atten_weight = atten_weights[i]
                        atten_weight = atten_weight.clip(torch.quantile(
                                        atten_weight, 0.1), torch.quantile(atten_weight, 0.9))
                        patch_atten_weights.append(atten_weight.clone())
                    patch_atten_weights = torch.stack(patch_atten_weights)

                patch_atten_weights /= patch_atten_weights.sum(dim=1, keepdims=True)

            patch_sim = torch.bmm(patch_emb_q, patch_atten_output.permute(0, 2, 1)) / self.local_temperature
            patch_num = patch_sim.size(1)
            patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
            targets = torch.arange(patch_num).type_as(patch_emb_q).long().repeat(bz)

            # loss_patch_1 = F.cross_entropy(patch_sim_1, targets)
            loss_patch_1 = torch.sum(F.cross_entropy(
                patch_sim_1, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
            loss_patch_2 = torch.sum(F.cross_entropy(
                patch_sim_2, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            loss_patch = (loss_patch_1 + loss_patch_2) / 2.

            loss_local = loss_patch + loss_word

        else:
            loss_local = loss_word

        return loss_local
        """ 