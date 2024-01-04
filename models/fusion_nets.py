import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple
from torchsummary import summary


def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class DotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """
    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()

    def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        return context, attn


############### Fusion ###################
class LinearFusion(nn.Module):
    def __init__(self, args):
        super(LinearFusion, self).__init__()
        dim = 256 + args.aux_feat_dim_per_granularity
        self.fc1 = nn.Linear(dim, args.fusion_final_dim) # change 512, 576, 640, 704, 768
        self.ln = nn.LayerNorm(args.aux_feat_dim_per_granularity) 

    def forward(self, img_features, sent_emb):
        #sent_emb = self.ln(sent_emb)
        concat_features =  torch.cat((img_features, sent_emb), dim=1)
        out = self.fc1(concat_features)
        return out 


class SelfAttention(nn.Module):
    def __init__(self, channel_dim, scale=2):
        super(SelfAttention, self).__init__()
        self.inplanes = channel_dim
        self.query_proj = nn.Conv2d(self.inplanes, self.inplanes // scale, 1)
        self.key_proj = nn.Conv2d(self.inplanes,  self.inplanes // scale, 1)
        self.value_proj = nn.Conv2d(self.inplanes, self.inplanes, 1)

        self.sqrt_dim = np.sqrt(channel_dim / scale)


    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        query = self.query_proj(y) # y--> text
        N,C,W,H = query.size()
        query = query.contiguous().view(N, C, H*W) #.transpose(2,1)

        key = self.key_proj(x) # x-->image
        key = key.contiguous().view(N, C, -1)
        key = key.transpose(2,1) #N, HW, C

        # compute attention
        attention = torch.bmm(key, query) / self.sqrt_dim 

        assert attention.size() == (N,H*W,H*W)
        attention = F.softmax(attention, dim=-1)

        # g transform
        value = self.value_proj(x) #x --> image
        N, C, W, H = y.size()
        value = value.contiguous().view(N, C, -1)
        value = value.transpose(2, 1) #N, HW, C
        
        # final response
        response = torch.bmm(attention, value)
        response = response.permute(0, 2, 1) #N, C, HW
        response = response.contiguous().view(N, C, W, H)
        return response
        


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 32, num_heads: int = 1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)  # BxQ_LENxNxD
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)      # BxK_LENxNxD
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)  # BxV_LENxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxV_LENxD

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # BxTxND
        return context




class Working_bad(nn.Module):
    def __init__(self, channel_dim):
        super(Working_bad,self).__init__()
        channel_dim = 144
        self.bn_img = nn.BatchNorm2d(channel_dim)
        self.bn_word = nn.BatchNorm2d(channel_dim)
        self.projection = nn.Linear(256, channel_dim)

        self.sa = SelfAttention(channel_dim, scale=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv = nn.Conv2d(256, channel_dim, kernel_size=(3, 3), padding=0)
        self.relu = nn.ReLU()
        self.ln_1 = nn.LayerNorm([channel_dim, 12, 12])
        self.ln_2 = nn.LayerNorm([channel_dim, 6, 6])

        self.ln_gl_image = nn.LayerNorm([256])
        self.ln_sent = nn.LayerNorm([256])
        self.linear = nn.Linear(1296, 512)


    def forward(self, img, word, gl_img, sent):
        img = self.relu(self.conv(img))
        img = self.bn_img(img)
        #img = F.normalize(img, p=2, dim=1)

        word = self.projection(word.transpose(1, 2)) #cap_len, 64
        word = torch.bmm(word.transpose(1, 2), word) / np.sqrt(144) #batch x 256 x 256
        word = word.unsqueeze(-1).view(word.size(0), word.size(1), 12, 12)
        word = self.bn_word(word)
        #word = F.normalize(word, p=2, dim=1)

        #img = self.sa(img, img)
        #iw = self.ln(img)
        iw = self.sa(img, word) #img, word
        iw = self.ln_1(iw)
        iw = self.maxpool(iw)

        iw = self.sa(iw, iw) #img, word
        iw = self.ln_2(iw)
        iw = self.maxpool(iw)

        iw = iw.view(iw.size(0), -1) #batch_size x 1024

        #img_sent = self.pl_cfa(gl_img, sent)
        #iw = torch.cat((iw, img_sent), dim=1)
        iw = self.linear(iw)
        #gl_img = self.ln_gl_image(gl_img) 
        #sent = self.ln_sent(sent)
        #return torch.concat((iw, gl_img, sent), dim=1) 
        return iw 



class Working(nn.Module):
    def __init__(self, channel_dim):
        super(Working,self).__init__()
        channel_dim = 36
        self.bn_img = nn.BatchNorm2d(channel_dim)
        self.bn_word = nn.BatchNorm2d(channel_dim)
        self.projection = nn.Linear(256, channel_dim)

        self.sa = SelfAttention(channel_dim, scale=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv = nn.Conv2d(256, channel_dim, kernel_size=(3, 3), padding=0)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm([channel_dim, 6, 6])
        self.ln_gl_image = nn.LayerNorm([256])
        self.ln_sent = nn.LayerNorm([256])
        self.linear = nn.Linear(324, 128)


    def forward(self, img, word, gl_img, sent):
        img = self.maxpool(self.relu(self.conv(img)))
        img = self.bn_img(img)
        #img = F.normalize(img, p=2, dim=1)

        word = self.projection(word.transpose(1, 2)) #cap_len, 64
        word = torch.bmm(word.transpose(1, 2), word) / np.sqrt(36) #batch x 256 x 256
        word = word.unsqueeze(-1).view(word.size(0), word.size(1), 6, 6)
        word = self.bn_word(word)
        #word = F.normalize(word, p=2, dim=1)

        #img = self.sa(img, img)
        #iw = self.ln(img)
        iw = self.sa(img, word) #img, word
        iw = self.ln(iw)
        iw = self.maxpool(iw)
        iw = iw.view(iw.size(0), -1) #batch_size x 1024

        #img_sent = self.pl_cfa(gl_img, sent)
        #iw = torch.cat((iw, img_sent), dim=1)
        iw = self.linear(iw)
        gl_img = self.ln_gl_image(gl_img) 
        sent = self.ln_sent(sent)
        return torch.concat((iw, gl_img, sent), dim=1) 


# Doesn't produce good resutls
class WordLevelCFA_LSTM(nn.Module):
    def __init__(self, channel_dim, scale=2):
        super(WordLevelCFA_LSTM,self).__init__()
        self.channel_dim = channel_dim
        self.bn_img = nn.BatchNorm2d(channel_dim)
        self.sa = SelfAttention(channel_dim, scale)
        self.avg_pool = nn.AvgPool2d(kernel_size = 8)
        self.conv = nn.Conv2d(256, channel_dim, kernel_size=(3, 3), padding=2)
        self.relu = nn.ReLU()

        self.ln1 = nn.LayerNorm([256, 16, 16])
        self.ln2 = nn.LayerNorm([256, 16, 16])
        self.linear = nn.Linear(1024, 768)
        

    def forward(self, gl_img: Tensor, word: Tensor) -> Tensor:
        img = torch.zeros((gl_img.size(0), 256, 16, 16), device="cuda")
        img[:, :, 1:15, 1:15] = gl_img 

        word = torch.bmm(word, word.transpose(1, 2)) / np.sqrt(self.channel_dim) #batch x 64 x 64
        word = word.unsqueeze(-1).view(word.size(0), word.size(1), 16, 16)
        
        img = self.sa(img, img)
        iw = self.ln1(img)
        iw = self.sa(img, word) #img, word

        #iw = self.avg_pool(iw)
        iw = self.ln2(iw)
        iw = self.avg_pool(iw)
        iw = iw.view(iw.size(0), -1) #batch_size x 1024
        iw = self.linear(iw)
        return iw 



class ParagraphLevelCFA(nn.Module):
    def __init__(self):
        super(ParagraphLevelCFA, self).__init__()

        self.mha = torch.nn.MultiheadAttention(embed_dim = 128, num_heads = 1, dropout=0.1, batch_first=True) #64
        self.linear_project = torch.nn.Linear(768, 128)
        self.ln = nn.LayerNorm(64)

    def forward(self, img: Tensor, sent_emb: Tensor) -> Tensor:
        bs = img.size(0)
        img = img.contiguous().view(bs, 8, 64)  #8, 64
        sent_emb = sent_emb.contiguous().view(bs, 1, 64)  #1, 64

        sent_feats = self.mha(sent_emb, img, img)
        sent_feats = sent_feats[0].contiguous().view(bs, -1) #batch_size x 64
        self.ln(sent_feats)
        return sent_feats  



class ConcatAttention(nn.Module):
    def __init__(self):
        super(ConcatAttention, self).__init__()
        self.mha = torch.nn.MultiheadAttention(embed_dim=256, num_heads = 1, dropout=0.2, batch_first=True)
        self.linear = torch.nn.Linear(768, 512)

    def forward(self, img: Tensor, sent_emb: Tensor) -> Tensor:
        bs = img.size(0) 
        patch = torch.cat((img, sent_emb), dim = 1)
        patch = patch.contiguous().view(bs, 3, 256)
        patch = self.mha(patch, patch, patch)
        patch = patch[0].contiguous().view(bs, -1) 
        return self.linear(patch) 


if __name__ == "__main__":
    bs = 16
    img = torch.randn((bs, 256, 14, 14))
    word_emb = torch.randn((bs, 256, 23))
    w = Working(channel_dim = 256)
    a = w(img, word_emb)
    print(a.shape)