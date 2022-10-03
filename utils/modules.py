import os
import os.path as osp
from scipy import linalg
from sklearn import metrics
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import mkdir_p
from utils.test_dataset import prepare_test_data
from tqdm import tqdm 


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

    print("\nAUC {:.4f} | EER {:.4f}".format(auc, eer))
    return auc, eer 


def test(test_dl, model, netG, text_encoder, args):
    device = args.device
    netG = netG.eval()
    preds = []
    labels = []

    loop = tqdm(total=len(test_dl))
    for step, data in enumerate(test_dl, 0):
        img1, img2, sent_emb1, sent_emb2, pair_label  = prepare_test_data(data, text_encoder)
        img1 = img1.to(device).requires_grad_()
        img2 = img2.to(device).requires_grad_()

        sent_emb1 = sent_emb1.to(device).requires_grad_()
        sent_emb2 = sent_emb2.to(device).requires_grad_()
        pair_label = pair_label.to(device)

        img_features = get_features(model, img1)
        out1 = netG(img_features, sent_emb1)

        img_features = get_features(model, img2)
        out2 = netG(img_features, sent_emb2)

        del img_features
        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        pred = cosine_sim(out1, out2)
        preds += pred.data.cpu().tolist()
        labels += pair_label.data.cpu().tolist()

        # update loop information
        loop.update(1)
        loop.set_postfix()

    loop.close()
    best_acc, best_th = cal_accuracy(preds, labels)
    calculate_scores(preds, labels)
    print("accuracy: %0.4f; threshold %0.4f" %(best_acc, best_th))


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
