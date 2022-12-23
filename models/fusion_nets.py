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


class AdditiveAttention(nn.Module):
    """
     Applies a additive attention (bahdanau) mechanism on the output features from the decoder.
     Additive attention proposed in "Neural Machine Translation by Jointly Learning to Align and Translate" paper.

     Args:
         hidden_dim (int): dimesion of hidden state vector

     Inputs: query, value
         - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
         - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.

     Returns: context, attn
         - **context**: tensor containing the context vector from attention mechanism.
         - **attn**: tensor containing the alignment from the encoder outputs.

     Reference:
         - **Neural Machine Translation by Jointly Learning to Align and Translate**: https://arxiv.org/abs/1409.0473
    """
    def __init__(self, hidden_dim: int) -> None:
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)
        return context, attn



class MultiHeadLocationAwareAttention(nn.Module):
    """
    Applies a multi-headed location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    In the above paper applied a signle head, but we applied multi head concept.

    Args:
        hidden_dim (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        conv_out_channel (int): The number of out channel in convolution

    Inputs: query, value, prev_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **prev_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s attention (alignment)

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoder outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.

    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, conv_out_channel: int = 10) -> None:
        super(MultiHeadLocationAwareAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = int(hidden_dim / num_heads)
        self.conv1d = nn.Conv1d(num_heads, conv_out_channel, kernel_size=3, padding=1)
        self.loc_proj = nn.Linear(conv_out_channel, self.dim, bias=False)
        self.query_proj = nn.Linear(hidden_dim, self.dim * num_heads, bias=False)
        self.value_proj = nn.Linear(hidden_dim, self.dim * num_heads, bias=False)
        self.score_proj = nn.Linear(self.dim, 1, bias=True)
        self.bias = nn.Parameter(torch.rand(self.dim).uniform_(-0.1, 0.1))

    def forward(self, query: Tensor, value: Tensor, last_attn: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len = value.size(0), value.size(1)

        if last_attn is None:
            last_attn = value.new_zeros(batch_size, self.num_heads, seq_len)

        loc_energy = torch.tanh(self.loc_proj(self.conv1d(last_attn).transpose(1, 2)))
        loc_energy = loc_energy.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(-1, seq_len, self.dim)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1, 3)
        query = query.contiguous().view(-1, 1, self.dim)
        value = value.contiguous().view(-1, seq_len, self.dim)

        score = self.score_proj(torch.tanh(value + query + loc_energy + self.bias)).squeeze(2)
        attn = F.softmax(score, dim=1)

        value = value.view(batch_size, seq_len, self.num_heads, self.dim).permute(0, 2, 1, 3)
        value = value.contiguous().view(-1, seq_len, self.dim)

        context = torch.bmm(attn.unsqueeze(1), value).view(batch_size, -1, self.num_heads * self.dim)
        attn = attn.view(batch_size, self.num_heads, -1)

        return context, attn


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._compute_relative_positional_encoding(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _compute_relative_positional_encoding(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score




############### Fusion ###################
class ConcatFusion(nn.Module):
    def __init__(self,):
        super(ConcatFusion, self).__init__()

    def forward(self, img_features, text_embedding):
        out = torch.cat((img_features, text_embedding), dim=1)
        return out 


class LinearFusion(nn.Module):
    def __init__(self, final_dim):
        super(LinearFusion, self).__init__()
        self.fc1 = nn.Linear(768, final_dim) # change 512, 576, 640, 704, 768
        self.ln = nn.LayerNorm(256)

    def forward(self, img_features, word_vector):
        word_vector = self.ln(word_vector)
        concat_features =  torch.cat((img_features, word_vector), dim=1) #sent_emb1, word_vector1
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



class WordLevelCFA(nn.Module):
    def __init__(self, channel_dim, scale=2):
        super(WordLevelCFA,self).__init__()
        self.channel_dim = channel_dim
        self.bn_img = nn.BatchNorm2d(channel_dim)
        self.sa = SelfAttention(channel_dim, scale)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv = nn.Conv2d(256, channel_dim, kernel_size=(3, 3), padding=1) #padding 2 for adaface
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size = 2)
        self.ln = nn.LayerNorm([64, 8, 8])
        #self.ln2 = nn.LayerNorm([64, 8, 8])
        self.pl_cfa = ParagraphLevelCFA()
        self.linear = nn.Linear(1024, 768)
        

    def forward(self, img: Tensor, word: Tensor) -> Tensor:
        img = self.maxpool(self.relu(self.conv(img)))
        img = self.bn_img(img)
        word = torch.bmm(word, word.transpose(1, 2)) / np.sqrt(self.channel_dim) #batch x 64 x 64
        word = word.unsqueeze(-1).view(word.size(0), word.size(1), 8, 8)

        img = self.sa(img, img)
        iw = self.ln(img)
        iw = self.sa(img, word) #img, word
        #iw = self.ln2(iw)
        iw = self.maxpool(iw)
        iw = iw.view(iw.size(0), -1) #batch_size x 1024

        #img_sent = self.pl_cfa(gl_img, sent)
        #iw = torch.cat((iw, img_sent), dim=1)
        iw = self.linear(iw)
        return iw 



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
        #self.ln = nn.LayerNorm(64)

    def forward(self, img: Tensor, sent_emb: Tensor) -> Tensor:
        bs = img.size(0)
        img = img.contiguous().view(bs, 8, 64)  #8, 64
        sent_emb = sent_emb.contiguous().view(bs, 1, 64)  #1, 64

        sent_feats = self.mha(sent_emb, img, img)
        sent_feats = sent_feats[0].contiguous().view(bs, -1) #batch_size x 64
        #self.ln(sent_feats)
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
    word_emb = torch.randn((bs, 256, 18))
    w = WordLevelCFA_LSTM(channel_dim = 64)
    a = w(img, word_emb, 0, 0)
    print(a.shape)