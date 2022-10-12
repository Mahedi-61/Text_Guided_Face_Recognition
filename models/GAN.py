import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary


class NetG1(nn.Module):
    def __init__(self, ngf, nz, cond_dim, imsize, ch_size):
        super(NetG1, self).__init__()
        self.ngf = ngf
        # input noise (batch_size, 100)
        self.fc = nn.Linear(nz, ngf*8*4*4)
        # build GBlocks
        self.GBlocks = nn.ModuleList([])
        in_out_pairs = get_G_in_out_chs(ngf, imsize)
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.GBlocks.append(G_Block(cond_dim+nz, in_ch, out_ch, upsample=True))
        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(out_ch, ch_size, 3, 1, 1),
            nn.Tanh(),
            )

    def forward(self, noise, c): # x=noise, c=ent_emb
        # concat noise and sentence
        out = self.fc(noise)
        out = out.view(noise.size(0), 8*self.ngf, 4, 4)
        cond = torch.cat((noise, c), dim=1)
        # fuse text and visual features
        for GBlock in self.GBlocks:
            out = GBlock(out, cond)
        # convert to RGB image
        out = self.to_rgb(out)
        return out


class NetG(nn.Module):
    def __init__(self, num_classes):
        super(NetG, self).__init__()
        self.fc1 = nn.Linear(1280, 1024)
        #self.relu1 = nn.ReLU(inplace=True)
        #self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, img_features, cond):
        concat_features = torch.cat((img_features, cond), dim=1)
        out = self.fc1(concat_features)
        #out = self.relu1(out)
        #out = self.fc2(out)
        return out 


class G_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, upsample):
        super(G_Block, self).__init__()
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch 
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.fuse1 = DFBLK(cond_dim, in_ch)
        self.fuse2 = DFBLK(cond_dim, out_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, y):
        h = self.fuse1(h, y)
        h = self.c1(h)
        h = self.fuse2(h, y)
        h = self.c2(h)
        return h

    def forward(self, x, y):
        if self.upsample==True:
            x = F.interpolate(x, scale_factor=2)
        return self.shortcut(x) + self.residual(x, y)



class DFBLK(nn.Module):
    def __init__(self, cond_dim, in_ch):
        super(DFBLK, self).__init__()
        self.affine0 = Affine(cond_dim, in_ch)
        self.affine1 = Affine(cond_dim, in_ch)

    def forward(self, x, y=None):
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        return h


class Affine(nn.Module):
    def __init__(self, cond_dim, num_features):
        super(Affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)        

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias



def get_G_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs



class MHSA(nn.Module):
  def __init__(self,
         emb_dim,
         kqv_dim,
         num_heads=1):
    super(MHSA, self).__init__()
    self.emb_dim = emb_dim
    self.kqv_dim = kqv_dim
    self.num_heads = num_heads

    self.w_k = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
    self.w_q = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
    self.w_v = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
    self.w_out = nn.Linear(kqv_dim * num_heads, emb_dim)

  def forward(self, x):

    b, t, _ = x.shape
    e = self.kqv_dim
    h = self.num_heads
    keys = self.w_k(x).view(b, t, h, e)
    values = self.w_v(x).view(b, t, h, e)
    queries = self.w_q(x).view(b, t, h, e)

    keys = keys.transpose(2, 1)
    queries = queries.transpose(2, 1)
    values = values.transpose(2, 1)

    dot = queries @ keys.transpose(3, 2)
    dot = dot / np.sqrt(e)
    dot = F.softmax(dot, dim=3)

    out = dot @ values
    out = out.transpose(1,2).contiguous().view(b, t, h * e)
    out = self.w_out(out)
    return out

if __name__ == "__main__":
    print("Bismillah")
    model = NetG(num_classes=200)
    summary(model,  [(1024,), (256,)], device='cpu')