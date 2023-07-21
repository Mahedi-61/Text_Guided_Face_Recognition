import sys
import os.path as osp
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm 
from torchvision import transforms
import os 

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils.prepare import prepare_test_data, prepare_test_data_Bert
from utils.dataset_utils import get_imgs

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
    gl_feats, lc_feats = model(imgs)
    return gl_feats, lc_feats 


def get_features_adaface(model, imgs):
    img_features, word_features, norm = model(imgs)
    return img_features, word_features, norm 


def get_tpr(fprs, tprs):
    fpr_val = [10 ** -5, 10 ** -4, 10 ** -3]
    tpr_fpr_row = []
    for fpr_iter in np.arange(len(fpr_val)):
        _, min_index = min(
            list(zip(abs(fprs - fpr_val[fpr_iter]), range(len(fprs)))))
        tpr_fpr_row.append(tprs[min_index] * 100)
    return tpr_fpr_row



################## scores ####################
def calculate_scores(y_score, y_true, args):
    # sklearn always takes (y_true, y_pred)
    fprs, tprs, threshold = metrics.roc_curve(y_true, y_score)
 
    fprs = np.flipud(fprs)
    tprs = np.flipud(tprs)

    eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]
    auc = metrics.auc(fprs, tprs)
    tpr_fpr_row = get_tpr(fprs, tprs)

    print("AUC {:.4f} | EER {:.4f} | TPR@FPR=1e-5 {:.4f} | TPR@FPR=1e-4 {:.4f} | TPR@FPR=1e-3 {:.4f}".
        format(auc, eer, tpr_fpr_row[0], tpr_fpr_row[1], tpr_fpr_row[2]))

    if args.is_roc == True:
        data_dir = "."
        with open(os.path.join(data_dir, args.roc_file + '.npy'), 'wb') as f:
            np.save(f, y_true)
            np.save(f, y_score)



def calculate_identification_acc(y_score, args):
    with open(os.path.join(args.checkpoints_path, "concat.npy"), "wb") as f:
        np.save(f, y_score)

    total_number = int(np.sqrt(len(y_score)))
    print("total subjects: ", total_number)

    y_score = np.array(y_score).reshape((total_number, total_number))
    y_score = np.argmax(y_score, axis=1)
    y_true = np.arange(total_number)
    acc = sum([1 for i, j in zip(y_score, y_true) if i==j])

    print("identification accuracy (%)", (acc/total_number) * 100)



def test(test_dl, model, net, text_encoder, text_head, args):
    device = args.device
    net.eval()
    text_encoder.eval()
    if args.using_BERT == True: text_head.eval()
    preds = []
    labels = []

    loop = tqdm(total=len(test_dl))
    for step, data in enumerate(test_dl, 0):
        if args.using_BERT == True: 
            img1, img2, words_emb1, words_emb2, word_vector1, word_vector2, sent_emb1, sent_emb2, \
                pair_label = prepare_test_data_Bert(data, text_encoder, text_head)

        elif args.using_BERT == False:
            img1, img2, words_emb1, words_emb2, sent_emb1, sent_emb2, pair_label = prepare_test_data(data, text_encoder)
        
        # upload to cuda
        img1 = img1.to(device).requires_grad_()
        img2 = img2.to(device).requires_grad_()
        pair_label = pair_label.to(device)

        # get global and local image features from COTS model
        if args.model_type == "arcface":
            global_feat1,  local_feat1 = get_features(model, img1)
            global_feat2,  local_feat2 = get_features(model, img2)

        elif args.model_type == "adaface":
            global_feat1,  local_feat1, norm = get_features_adaface(model, img1)
            global_feat2,  local_feat2, norm = get_features_adaface(model, img2)

        sent_emb1 = sent_emb1.to(device)
        sent_emb2 = sent_emb2.to(device)

        # sentence & word featurs 
        if args.fusion_type == "concat":
            if args.using_BERT == False:
                out1 =  torch.cat((global_feat1, sent_emb1), dim=1) 
                out2 =  torch.cat((global_feat2, sent_emb2), dim=1)

            elif args.using_BERT == True:
                out1 =  torch.cat((global_feat1, word_vector1), dim=1) #sent_emb1
                out2 =  torch.cat((global_feat2, word_vector2), dim=1) #sent_emb2

        elif args.fusion_type == "linear":
            if args.using_BERT == True:
                out1 =  net(global_feat1, word_vector1)
                out2 =  net(global_feat2, word_vector2)

            elif args.using_BERT == False:
                out1 =  net(global_feat1, sent_emb1)
                out2 =  net(global_feat2, sent_emb2)

        elif args.fusion_type == "concat_attention":
            out1 =  net(global_feat1, sent_emb1)
            out2 =  net(global_feat2, sent_emb2)

        elif args.fusion_type == "paragraph_attention":
            out1 =  net(global_feat1,  sent_emb1)
            out2 =  net(global_feat2,  sent_emb2)

        elif args.fusion_type == "cross_attention":
            words_emb1 = words_emb1.to(device).requires_grad_()
            words_emb2 = words_emb2.to(device).requires_grad_()

            out1 = net(local_feat1, words_emb1)
            out2 = net(local_feat2, words_emb2)

        del local_feat1, local_feat2, words_emb1, words_emb2

        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        pred = cosine_sim(out1, out2)
        preds += pred.data.cpu().tolist()
        labels += pair_label.data.cpu().tolist()

        # update loop information
        loop.update(1)
        loop.set_postfix()

    loop.close()
    #best_acc, best_th = cal_accuracy(preds, labels)
    if not args.is_ident: 
        calculate_scores(preds, labels, args)
    else:
        calculate_identification_acc(preds, args)
    
    #print("accuracy: %0.4f; threshold %0.4f" %(best_acc, best_th))



############
#for identification
def get_img_feactures_dict(model, args):
    with open(args.test_pair_list, 'r') as fd:
        pairs = fd.readlines()

    ls_img = list(set([pair.split(" ")[0] for pair in pairs]))
    ls_img += list(set([pair.split(" ")[1] for pair in pairs]))
    imgs_path = [os.path.join(args.data_dir, args.dataset_name, "test_images", i) for i in ls_img]
    imsize = 128 if args.model_type == "arcface" else 112
    image_transform = transforms.Compose([transforms.RandomCrop(imsize)])
    img_features_dict = {}

    for img_path in imgs_path:
        img = get_imgs(img_path, args.split, transform = image_transform, model_type = args.model_type).to(args.device).requires_grad_()
        img_features_dict[img_path] = [get_features(model, img.unsqueeze(0))[0]]

    return img_features_dict