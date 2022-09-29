import os
import os.path as osp
from scipy import linalg
from sklearn import metrics
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from utils.utils import mkdir_p

from utils.train_dataset import prepare_train_data
from utils.test_dataset import prepare_test_data


############   modules   ############
def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0

    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def get_features(model, imgs):
    img_features = model(imgs)
    flip_imgs = torch.squeeze(imgs, 1)
    a = torch.stack([torch.fliplr(flip_imgs[i]) for i in range(0, flip_imgs.size(0))])
    flip_img_features = model(a.unsqueeze(dim=1))
    img_features = torch.cat((img_features, flip_img_features), dim=1)
    del flip_img_features
    return img_features


def calculate_scores(y_score, y_true):
    # sklearn always takes (y_true, y_pred)
    fprs, tprs, threshold = metrics.roc_curve(y_true, y_score)
    eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]
    auc = metrics.auc(fprs, tprs)

    print("AUC {:.4f} | EER {:.4f}".format(auc, eer))
    return auc, eer 


def train(train_dl, model, netG, text_encoder, optimizerG, args):
    device = args.device
    epoch = args.current_epoch
    max_epoch = args.max_epoch
    netG = netG.train()

    loop = tqdm(total=len(train_dl))
    criterion = torch.nn.CrossEntropyLoss()

    for step, data in enumerate(train_dl, 0):
        imgs, sent_emb, words_embs, keys, label = prepare_train_data(data, text_encoder)
        imgs = imgs.to(device).requires_grad_()
        sent_emb = sent_emb.to(device).requires_grad_()
        #words_embs = words_embs.to(device).requires_grad_()
        label = label.to(device)
        
        img_features = get_features(model, imgs)
        output = netG(img_features, sent_emb)
        
        loss = criterion(output, label)
        optimizerG.zero_grad()
        loss.backward()
        optimizerG.step()

        # update loop information
        loop.update(1)
        loop.set_description(f'Training Epoch [{epoch}/{max_epoch}]')
        loop.set_postfix()

    loop.close()
    del img_features


def test(test_dl, model, netG, text_encoder, args):
    device = args.device
    netG = netG.eval()
    preds = []
    labels = []
    for step, data in enumerate(test_dl, 0):
        img1, img2, sent_emb1, sent_emb2, pair_label  = prepare_test_data(data, text_encoder)
        img1 = img1.to(device).requires_grad_()
        img2 = img2.to(device).requires_grad_()

        sent_emb1 = sent_emb1.to(device).requires_grad_()
        sent_emb2 = sent_emb2.to(device).requires_grad_()
        pair_label = pair_label.to(device)

        img_features = get_features(model, img1)
        cobmined_feat = torch.cat((img_features, sent_emb1), dim=1)
        out1 = netG(cobmined_feat)

        img_features = get_features(model, img2)
        cobmined_feat = torch.cat((img_features, sent_emb2), dim=1)
        out2 = netG(cobmined_feat)

        del img_features, cobmined_feat
        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        pred = cosine_sim(out1, out2)
        preds += pred.data.cpu().tolist()
        labels += pair_label.data.cpu().tolist()

    best_acc, best_th = cal_accuracy(preds, labels)
    print("accuracy: ", best_acc)
    calculate_scores(preds, labels)


def save_single_imgs(imgs, save_dir, time, dl_len, batch_n, batch_size):
    for j in range(batch_size):
        folder = save_dir
        if not os.path.isdir(folder):
            #print('Make a new folder: ', folder)
            mkdir_p(folder)
        im = imgs[j].data.cpu().numpy()
        # [-1, 1] --> [0, 255]
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        filename = 'imgs_n%06d_gpu%1d.png'%(time*dl_len+batch_size*batch_n+j, get_rank())
        fullpath = osp.join(folder, filename)
        im.save(fullpath)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def predict_loss(predictor, img_feature, text_feature, negtive):
    output = predictor(img_feature, text_feature)
    err = hinge_loss(output, negtive)
    return output,err


def hinge_loss(output, negtive):
    if negtive==False:
        err = torch.nn.ReLU()(1.0 - output).mean()
    else:
        err = torch.nn.ReLU()(1.0 + output).mean()
    return err


def logit_loss(output, negtive):
    batch_size = output.size(0)
    real_labels = torch.FloatTensor(batch_size,1).fill_(1).to(output.device)
    fake_labels = torch.FloatTensor(batch_size,1).fill_(0).to(output.device)
    output = nn.Sigmoid()(output)

    if negtive==False:
        err = nn.BCELoss()(output, real_labels)
    else:
        err = nn.BCELoss()(output, fake_labels)
    return err
