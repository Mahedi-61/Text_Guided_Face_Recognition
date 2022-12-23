import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel
import torch.nn.functional as F
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


class ResNetFace(nn.Module):
    def __init__(self, block, layers, use_se=True):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNetFace, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc5 = nn.Linear(512 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        word_feat = x 

        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.bn5(x)

        return word_feat, x  


def resnet_face18(use_se=True, **kwargs):
    model = ResNetFace(IRBlock, [2, 2, 2, 2], use_se=use_se, **kwargs)
    return model



############### Encoder-Decoder ###################
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout = 0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        #self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        #self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        #x = self.gelu(projected)
        #x = self.fc(x)
        #x = self.dropout(x)
        #x = x + projected
        #x = self.layer_norm(x)
        return projected



class BERT_ENCODER(nn.Module):
    def __init__(self, args):
        super(BERT_ENCODER, self).__init__()
        self.model = BertModel.from_pretrained(args.bert_config)
        #self.model = AutoModel.from_pretrained(args.bert_config)
        for p in self.model.parameters():
            p.requires_grad = True

        print("Bert encoder trainable: ", True)

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
        Ks = [1, 2, 3]
        in_channel = 1
        out_channel = feat_dim #* 4
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, 768)) for K in Ks]) 

        self.dropout = nn.Dropout(0.1)
        #self.mapping = nn.Linear(out_channel, feat_dim)

    def forward(self, words_emb):
        x = words_emb.unsqueeze(1)  # (batch_size, 1, token_num, embedding_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(batch_size, out_channel, W), ...]*len(Ks)

        #remove these 2 lines to skip second projection
        #x = [self.dropout(i.transpose(2, 1)) for i in x] 
        #x = [self.mapping(i).transpose(2, 1) for i in x]
        return x



class BERTHeading(nn.Module):
    def __init__(self, args):
        super(BERTHeading, self).__init__()
        self.feat_dim = args.aux_feat_dim_per_granularity
        self.sentence_feat = ProjectionHead(embedding_dim=768, projection_dim=self.feat_dim)
        self.bwm = Bert_Word_Mapping(self.feat_dim)
        self.word_feat = nn.Linear(768, self.feat_dim)

    def get_each_word_feature(self, x):
        bs = x[0].size(0)
        a = x[0].transpose(2, 1)
        b = x[1].transpose(2, 1)
        c = x[2].transpose(2, 1)
        code = []
        for i in range(bs):
            t = [torch.amax(torch.stack((a[i, j], b[i, j], c[i, j])), dim=0) for j in range(18)]
            t +=  [torch.amax(torch.stack((a[i, 18], b[i, 18])), dim=0)]
            t += [torch.cuda.FloatTensor(a[i, 19])]  
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
        sent_emb = self.sentence_feat(sent_emb) #batch_size x 64
        #words_emb = self.word_feat(words_emb) #batch_size x 20 x 256
        
        x = self.bwm(words_emb)
        words_emb = self.get_each_word_feature(x) 
        word_vector = self.get_word_feature(x)

        words_emb = words_emb.transpose(1, 2)
        return words_emb, word_vector, sent_emb



class RNN_ENCODER(nn.Module):
    def __init__(self, args, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()

        self.n_steps = args.lstm_words_num
        self.ntoken = args.vocab_size  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = args.rnn_type
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
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
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
        if self.rnn_type == 'LSTM':
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
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb
        


class ResNet18_ArcFace_ENCODER(nn.Module):
    def __init__(self, args):
        super(ResNet18_ArcFace_ENCODER, self).__init__()

        model_path = "weights/celeba/FE/resnet18_celeba_110.pth"
        model = ResNetFace(IRBlock, [2, 2, 2, 2], use_se= False)
        
        weights = torch.load(model_path)
        state_dict = {
                key[7:]: value
                for key, value in weights.items()
            }
        model.load_state_dict(state_dict)
        print('Load pretrained ResNet18 ArcFace model from ', model_path)
        for p in model.parameters():
            p.requires_grad = False

        self.define_module(model)
        print("ArcFace Encoder trainable ", False)

    def define_module(self, model):
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.prelu = model.prelu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.bn4 = model.bn4
        self.dropout = model.dropout
        self.fc5 = model.fc5
        self.bn5 = model.bn5
        del model 
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = x #.permute(0, 2, 3, 1)

        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.bn5(x)

        return features, x



class ArcFace_Heading(nn.Module):
    def __init__(self, args):
        super(ArcFace_Heading, self).__init__()
        self.project_global = ProjectionHead(embedding_dim=512, projection_dim=args.aux_feat_dim_per_granularity)
        self.project_local =  ProjectionHead(embedding_dim=256, projection_dim=args.aux_feat_dim_per_granularity)
        
    def forward(self, local_image, global_image):
        local_image = local_image.permute((0, 2, 3, 1))
        local_image = self.project_local(local_image) #batch_size x 16 x 16 x 128
        local_image = F.normalize(local_image, p=2, dim=-1)
        local_image = local_image.permute((0, 3, 1, 2))

        global_image = self.project_global(global_image) #batch_size x 128
        global_image = F.normalize(global_image, p=2, dim=-1)
        return local_image, global_image 



class AdaFace(nn.Module):
    def __init__(self,
                 embedding_size,
                 classnum,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.kernel = torch.nn.Parameter(torch.Tensor(embedding_size, classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)


    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability
        del kernel_norm

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m



if __name__ == "__main__":
    from easydict import EasyDict as edict
    args = edict()