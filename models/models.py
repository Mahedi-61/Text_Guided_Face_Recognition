import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import (BertModel, AlignTextModel, CLIPTextModel, 
                          FlavaTextModel, BlipTextModel, GroupViTTextModel)
import torch.nn.functional as F
from models.fusion_nets import SelfAttention 
from torchsummary import summary
import numpy as np 
import math 


def l2_norm(input, axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output


def get_CLS_embedding(layer):
    return layer[:, 0, :]

############### Arc Face ##################### 
def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.PReLU(),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
        
class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out



############### Encoder-Decoder ###################
class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim,
        projection_dim,
        dropout = 0.4
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        #self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        x = self.projection(x)
        #x = self.gelu(projected)
        #x = self.fc(x)
        #x = self.dropout(x)
        #x = x + projected
        #x = self.layer_norm(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


def get_encoder(args):
    if args.bert_type == "bert":
        return BertModel.from_pretrained(args.bert_config)

    elif args.bert_type == "align":
        return AlignTextModel.from_pretrained(args.align_config)

    elif args.bert_type == "clip": #512 text dim
        return CLIPTextModel.from_pretrained(args.clip_config)
    
    elif args.bert_type == "blip":
        return BlipTextModel.from_pretrained(args.blip_config)
    
    elif args.bert_type == "falva":
        return FlavaTextModel.from_pretrained(args.falva_config)
    
    elif args.bert_type == "groupvit": #256 text dim
        return GroupViTTextModel.from_pretrained(args.groupvit_config)



class TextEncoder(nn.Module):
    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.model = get_encoder(args)

        print("Loading : ", args.bert_type)
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, captions, mask):
        outputs = self.model(captions, attention_mask=mask)

        ### Sentence features
        # outputs -> (last_hidden_state, pooler_output, hidden_states)
        # hidden_states -> tuple of lenght 13
        # another way to calculate sentece features
        # sent_feat = (word_feat * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)

        #embeddings = outputs[2][1:]
        #cls_embeddings = [get_CLS_embedding(embeddings[i]) for i in range(self.num_bert_layer)]
        #sent_emb = torch.mean(torch.stack(cls_embeddings, dim=1), dim=1) #batch_size x 768
        sent_emb = outputs[0][:,0,:]
        words_emb = outputs[0][:,1:,:] #outputs[0]
        return words_emb, sent_emb


class Bert_Word_Mapping(nn.Module):
    def __init__(self, feat_dim):
        super(Bert_Word_Mapping, self).__init__()
        Ks = [2, 3, 4]
        in_channel = 1
        out_channel = feat_dim #* 4
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, 768)) for K in Ks]) #512 for clip

        self.dropout = nn.Dropout(0.1)
        #self.mapping = nn.Linear(out_channel, feat_dim)

    def forward(self, words_emb):
        x = words_emb.unsqueeze(1)  # (batch_size, 1, token_num, embedding_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(batch_size, out_channel, W), ...]*len(Ks)
        return x


class TextHeading(nn.Module):
    def __init__(self, args):
        super(TextHeading, self).__init__()
        self.feat_dim = args.aux_feat_dim_per_granularity
        self.bwm = Bert_Word_Mapping(self.feat_dim)
        self.args = args 

        #self.sentence_feat = ProjectionHead(input_dim=768, projection_dim=self.feat_dim) #512 for clip, 256 for groupVit
        #self.word_feat = nn.Linear(768, self.feat_dim)

    def get_each_word_feature(self, x):
        bs = x[0].size(0)
        a = x[0].transpose(2, 1)
        b = x[1].transpose(2, 1)
        c = x[2].transpose(2, 1)
        code = []
        for i in range(bs):
            seq = self.args.bert_words_num - 1 - 3 #removing [CLS] token and two positions (for 1->2, 2->3)
            t = [torch.amax(torch.stack((a[i, j], b[i, j], c[i, j])), dim=0) for j in range(seq)]
            t +=  [torch.amax(torch.stack((a[i, seq], b[i, seq])), dim=0)]
            t += [torch.cuda.FloatTensor(a[i, seq+1])]  
            t = torch.stack(t)
            code.append(t)

        code = torch.stack(code)
        code = F.normalize(code, p=2, dim=2)
        return code 


    def get_word_feature(self, x):
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        output = torch.stack((x[0], x[1], x[2])).mean(dim=0)
        output = F.normalize(output, p=2, dim=1)
        return output 


    def forward(self, words_emb, sent_emb):
        #sent_emb = x = self.sentence_feat(sent_emb) #batch_size x 64
        #words_emb = self.word_feat(words_emb) #batch_size x 20 x 256
        
        x = self.bwm(words_emb)
        words_emb = self.get_each_word_feature(x) 
        sent_emb = self.get_word_feature(x)

        words_emb = words_emb.transpose(1, 2)
        return words_emb, sent_emb



class RNNEncoder(nn.Module):
    def __init__(self, args, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNNEncoder, self).__init__()

        self.n_steps = args.lstm_words_num
        self.ntoken = args.vocab_size  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # number of recurrent layers
        self.bidirectional = bidirectional
        self.en_type = args.en_type
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.en_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.en_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

        self.rnn.flatten_parameters() 


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.en_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))


        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.en_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)

        #normalize 
        sent_emb = F.normalize(sent_emb, p=2, dim=-1)
        return words_emb, sent_emb
        


class ImageHeading(nn.Module):
    def __init__(self, args):
        super(ImageHeading, self).__init__()
        self.project_global = ProjectionHead(input_dim=512, projection_dim=args.aux_feat_dim_per_granularity)
        self.imim = IMIM(args, channel_dim = 256)
        
    def forward(self, global_image, local_image):
        local_image = self.imim(local_image)
        global_image = self.project_global(global_image) #batch_size x 256

        return  global_image, local_image, 



class ArcFaceHeadingf(nn.Module):
    def __init__(self, args):
        super(ArcFaceHeadingf, self).__init__()
        channel_dim = args.aux_feat_dim_per_granularity * 2
        self.project_local =  ProjectionHead(embedding_dim=256, projection_dim=args.aux_feat_dim_per_granularity)
        self.project_global = ProjectionHead(embedding_dim=512*7*7, projection_dim=args.aux_feat_dim_per_granularity)

        self.bn_img = nn.BatchNorm2d(channel_dim)
        self.bn_1d = nn.BatchNorm1d(args.aux_feat_dim_per_granularity, affine=False)
        self.dropout = nn.Dropout(0.4)
        self.sa = SelfAttention(channel_dim, scale=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv = nn.Conv2d(512, channel_dim, kernel_size=(3, 3), padding=1) 
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm([channel_dim, 7, 7])
        #self.avg = nn.AvgPool2d(7)
        self.flat = nn.Flatten()

        
    def forward(self, global_image, local_image):
        #img = self.relu(self.conv(global_image)) #512x7x7
        img = global_image
        img = self.dropout(self.bn_img(img))
        img = self.sa(img, img)
        img = self.ln(img)
        img = self.flat(img)
        img = self.project_global(img)
        img = self.bn_1d(img)
        
        local_image = local_image.permute((0, 2, 3, 1))
        local_image = self.project_local(local_image) #batch_size x 16 x 16 x 128
        local_image = F.normalize(local_image, p=2, dim=-1)
        local_image = local_image.permute((0, 3, 1, 2))

        return img, local_image 



class IMIM(nn.Module):
    def __init__(self, args, channel_dim):
        super(IMIM, self).__init__()
        self.channel_dim = channel_dim
        self.project_local =  ProjectionHead(input_dim=256, 
                                             projection_dim=args.aux_feat_dim_per_granularity)
        self.bn_img = nn.BatchNorm2d(self.channel_dim)
        self.sa = SelfAttention(channel_dim = self.channel_dim, scale=1)
        self.conv1x1_1 = nn.Conv2d(self.channel_dim, self.channel_dim//2, kernel_size=(1, 1)) 
        self.relu = nn.ReLU()
        self.conv1x1_2 = nn.Conv2d(self.channel_dim//2, self.channel_dim, kernel_size=(1, 1)) 
        self.ln = nn.LayerNorm([self.channel_dim, 14, 14])
        
    def forward(self, img):
        img = self.bn_img(img)
        img = self.sa(img, img)
        img = self.ln(img)
    
        img = self.relu(self.conv1x1_1(img))
        img = self.relu(self.conv1x1_2(img))

        img = img.permute((0, 2, 3, 1))
        img = self.project_local(img) #batch_size x 14 x 14 x 256
        img = F.normalize(img, p=2, dim=-1)
        img = img.permute((0, 3, 1, 2))
        return img



if __name__ == "__main__":
    from easydict import EasyDict as edict
    args = edict()
    args.aux_feat_dim_per_granularity = 256
    x = torch.randn(128, 512, 14, 14)
    net = ImageHeading(args)
    y = net(x)
    print(y.shape)