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

        return x, word_feat 


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
            if args.is_second_step == False:
                trainable =  True 
            else:
                trainable = False
            p.requires_grad = trainable

        print("Bert encoder trainable: ", trainable)

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
    def __init__(self):
        super(Bert_Word_Mapping, self).__init__()
        Ks = [1, 2, 3]
        in_channel = 1
        out_channel = 64
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, 768)) for K in Ks]) 

    def forward(self, words_emb):
        x = words_emb.unsqueeze(1)  # (batch_size, 1, token_num, embedding_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(batch_size, out_channel, W), ...]*len(Ks)
        return x



class BERTHeading(nn.Module):
    def __init__(self, args):
        super(BERTHeading, self).__init__()
        self.sentence_feat = ProjectionHead(embedding_dim=768, projection_dim=64)
        self.word_feat = nn.Linear(768, 64)
        self.bwm = Bert_Word_Mapping()
        #self.dropout = nn.Dropout(0.1)
        #self.mapping = nn.Linear(out_channel, 256)

    def get_each_word_feature(self, x):
        bs = x[0].size(0)
        a = x[0].transpose(2, 1)
        b = x[1].transpose(2, 1)
        c = x[2].transpose(2, 1)
        code = []
        for i in range(bs):
            t = [torch.amax(torch.stack((a[i, j], b[i, j], c[i, j])), dim=0) for j in range(18)]
            t +=  [torch.amax(torch.stack((a[i, 18], b[i, 18])), dim=0)]
            t += [torch.cuda.FloatTensor(a[i, 19])]  #[torch.amax(torch.stack((a[i, 19], a[i, 19])), dim=0)]
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
        sent_emb = self.sentence_feat(sent_emb) #batch_size x 256
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
        #
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



class ResNet18_ArcFace_Heading(nn.Module):
    def __init__(self, args):
        super(ResNet18_ArcFace_Heading, self).__init__()
        self.project_global = ProjectionHead(embedding_dim=512, projection_dim=64)
        self.project_local =  ProjectionHead(embedding_dim=256, projection_dim=64)
        
    def forward(self, local_image, global_image):
        local_image = local_image.permute((0, 2, 3, 1))
        local_image = self.project_local(local_image) #batch_size x 16 x 16 x 128
        local_image = F.normalize(local_image, p=2, dim=-1)
        local_image = local_image.permute((0, 3, 1, 2))

        global_image = self.project_global(global_image) #batch_size x 128
        global_image = F.normalize(global_image, p=2, dim=-1)
        return local_image, global_image 



class CNN_ENCODER(nn.Module):
    def __init__(self, args):
        super(CNN_ENCODER, self).__init__()
        if args.using_BERT == True: self.nef = 768
        elif args.using_BERT == False:  self.nef = 256 

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))

        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.functional.interpolate(x,size=(299, 299), mode='bilinear', align_corners=False)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code


"""
class MANIGAN_Bert_Encoder():
    def __init__(self, args):
        super(MANIGAN_Bert_Encoder, self).__init__()
        self.args = args 
        self.model = BertModel.from_pretrained(self.args.bert_config, 
                                                output_hidden_states=True)
        for p in self.model.parameters():
            p.requires_grad = True
        self.model = nn.DataParallel(self.model, device_ids=args.gpu_id).cuda()

    def get_word_idx(self, sent: str, word: str):
        return sent.split(" ").index(word)

    def get_hidden_states(self, encoded, token_ids_word, model, layers):
        with torch.no_grad():
            output = model(**encoded)

            # Get all hidden states
            states = output.hidden_states

            # Stack and sum all requested layers
            output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

            # Only select the tokens that constitute the requested word
            word_tokens_output = output[token_ids_word]
            return word_tokens_output.mean(dim=0)


    def get_word_vector(self, sent, idx, tokenizer, model, layers):
        encoded = tokenizer.encode_plus(sent, add_special_tokens=True,
                            max_length = self.args.bert_words_num,
                            return_token_type_ids=False,
                            padding='max_length', 
                            return_tensors="pt")

        # get all token idxs that belong to the word of interest
        token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
        return self.get_hidden_states(encoded, token_ids_word, model, layers)


    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        input_mask_expanded =input_mask_expanded.cuda()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def get_embeddings(self, captions, layers=None ):
        # Use last four layers by default
        layers = [-4, -3, -2, -1] if layers is None else layers
        tokenizer = BertTokenizerFast.from_pretrained(self.args.bert_config)
        

        # dense_model = models.Dense(in_features= 768, out_features=256, activation_function= nn.ReLU())
        # model = SentenceTransformer(modules=[word_embedding_model, dense_model])

        with torch.no_grad():
            batch_word_embeddings = []
            batch_sentence_embeddings = []

            for sent in captions:
                encoded_input = tokenizer(sent, padding='max_length', max_length = self.args.bert_words_num, truncation=True, return_tensors='pt')
                model_output = self.model(**encoded_input)
                sent_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                word_embeddings = []

                max_limit = self.args.bert_words_num - 1
                if len(sent.split()) >= max_limit:
                    for word in sent.split()[:max_limit]:
                        idx = self.get_word_idx(sent, word)
                        word_embedding = self.get_word_vector(sent, idx, tokenizer, self.model, layers)
                        word_embeddings.append(word_embedding)
                else:
                    for word in sent.split():
                        idx = self.get_word_idx(sent, word)
                        word_embedding = self.get_word_vector(sent, idx, tokenizer, self.model, layers)
                        word_embeddings.append(word_embedding)

                    word_embeddings += [word_embedding] * (max_limit - len(word_embeddings)) # fill the rest with the <end> token

                word_embeddings = torch.stack([we for we in word_embeddings])
                batch_word_embeddings.append(word_embeddings)
                batch_sentence_embeddings.append(sent_embeddings)

        batch_word_embeddings = pad_sequence(batch_word_embeddings, batch_first=True)
        #batch_word_embeddings = batch_word_embeddings.transpose(1, 2)

        batch_sentence_embeddings = torch.stack(batch_sentence_embeddings)
        batch_sentence_embeddings = batch_sentence_embeddings.transpose(0, 1).contiguous()
        batch_sentence_embeddings = batch_sentence_embeddings.view(-1, 768)

        return batch_word_embeddings, batch_sentence_embeddings
"""


if __name__ == "__main__":
    from easydict import EasyDict as edict
    args = edict()

    """
    bh = BERTHeading(args)
    words_embd = torch.randn(16, 20, 768)
    sent_embd = torch.randn(16, 768)
    words_emb, word_vector, sent_emb = bh(words_embd, sent_embd)
    print(words_emb.shape)
    """