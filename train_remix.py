# This script is modfied from https://github.com/binli123/dsmil-wsi/blob/master/train_tcga.py

import argparse
import copy
import logging
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from torch.autograd import Variable

from model import abmil, dsmil
from tools.utils import setup_logger

warnings.simplefilter('ignore')


def get_bag_feats_v2(feats, bag_label, args):
    if isinstance(feats, str):
        # if feats is a path, load it
        feats = feats.split(',')[0]
        feats = torch.Tensor(np.load(feats)).cuda()
    
    feats = feats[np.random.permutation(len(feats))]
    if args.num_classes != 1:
        # mannual one-hot encoding, following dsmil
        label = np.zeros(args.num_classes)
        if int(bag_label) <= (len(label) - 1):
            label[int(bag_label)] = 1
        bag_label = Variable(torch.FloatTensor([label]).cuda())
        
    return bag_label, feats


def convert_label(labels, num_classes=2):
    # one-hot encoding for multi-class labels
    if num_classes > 1:
        # one-hot encoding
        converted_labels = np.zeros((len(labels), num_classes))
        for ix in range(len(labels)):
            converted_labels[ix, int(labels[ix])] = 1
        return converted_labels
    else:
        # return binary labels
        return labels


def inverse_convert_label(labels):
    # one-hot decoding
    if len(np.shape(labels)) == 1:
        return labels
    else:
        converted_labels = np.zeros(len(labels))
        for ix in range(len(labels)):
            converted_labels[ix] = np.argmax(labels[ix])
        return converted_labels


def mix_aug(src_feats, tgt_feats, mode='replace', rate=0.3, strength=0.5, shift=None):
    assert mode in ['replace', 'append', 'interpolate', 'cov', 'joint']
    auged_feats = [_ for _ in src_feats.reshape(-1, 512)]
    tgt_feats = tgt_feats.reshape(-1, 512)
    closest_idxs = np.argmin(cdist(src_feats.reshape(-1, 512), tgt_feats), axis=1)
    if mode != 'joint':
        for ix in range(len(src_feats)):
            if np.random.rand() <= rate:
                if mode == 'replace':
                    auged_feats[ix] = tgt_feats[closest_idxs[ix]]
                elif mode == 'append':
                    auged_feats.append(tgt_feats[closest_idxs[ix]])
                elif mode == 'interpolate':
                    generated = (1 - strength) * auged_feats[ix] + strength * tgt_feats[closest_idxs[ix]]
                    auged_feats.append(generated)
                elif mode == 'cov':
                    generated = auged_feats[ix][np.newaxis, :] + strength * shift[closest_idxs[ix]][np.random.choice(200, 1)]
                    auged_feats.append(generated.flatten())
                else:
                    raise NotImplementedError
    else:
        for ix in range(len(src_feats)):
            if np.random.rand() <= rate:
                # replace
                auged_feats[ix] = tgt_feats[closest_idxs[ix]]
            if np.random.rand() <= rate:
                # append
                auged_feats.append(tgt_feats[closest_idxs[ix]])
            if np.random.rand() <= rate:
                # interpolate
                generated = (1 - strength) * auged_feats[ix] + strength * tgt_feats[closest_idxs[ix]]
                auged_feats.append(generated)
            if np.random.rand() <= rate:
                # covary
                generated = auged_feats[ix][np.newaxis, :] + strength * shift[closest_idxs[ix]][np.random.choice(200, 1)]
                auged_feats.append(generated.flatten())
    return np.array(auged_feats)


def mix_the_bag_aug(bag_feats, idx, train_feats, train_labels, args, semantic_shifts=None):
    if args.mode is not None:
        # randomly select one bag from the same class
        positive_idxs = np.argwhere(train_labels.cpu().numpy() == train_labels[idx].item()).reshape(-1)
        selected_id = np.random.choice(positive_idxs)
        # lambda parameter
        strength = np.random.uniform(0, 1)
        bag_feats = mix_aug(bag_feats.cpu().numpy(), train_feats[selected_id].cpu().numpy(),
                            shift=semantic_shifts[selected_id] if args.mode == 'joint' or args.mode == 'cov' else None,
                            rate=args.rate, strength=strength, mode=args.mode)
        bag_feats = torch.Tensor([bag_feats]).cuda()
    bag_feats = bag_feats.view(-1, args.feats_size)
    return bag_feats


def train(train_feats, train_labels, milnet, criterion, optimizer, args, semantic_shifts=None):
    milnet.train()
    total_loss = 0
    for i in range(len(train_feats)):
        optimizer.zero_grad()
        bag_label, bag_feats = get_bag_feats_v2(train_feats[i], train_labels[i], args)
        # abort invalid features
        if torch.isnan(bag_feats).sum() > 0:
            continue
        bag_feats = mix_the_bag_aug(bag_feats, i, train_feats, train_labels, args, semantic_shifts)
        if args.model == 'dsmil':
            # refer to dsmil code
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5 * bag_loss + 0.5 * max_loss
        elif args.model == 'abmil':
            bag_prediction = milnet(bag_feats)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            loss = bag_loss
        else:
            raise NotImplementedError
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_feats), loss.item()))
    sys.stdout.write('\n')
    return total_loss / len(train_feats)


def test(test_feats, test_gts, milnet, criterion, args):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for i in range(len(test_feats)):
            bag_label, bag_feats = get_bag_feats_v2(test_feats[i], test_gts[i], args)
            bag_feats = bag_feats.view(-1, args.feats_size)
            if args.model == 'dsmil':
                ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
                max_prediction, _ = torch.max(ins_prediction, 0)
                bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
                loss = 0.5 * bag_loss + 0.5 * max_loss
            elif args.model == 'abmil':
                bag_prediction = milnet(bag_feats)
                bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                loss = bag_loss
            else:
                raise NotImplementedError
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_feats), loss.item()))
            test_labels.extend([bag_label.cpu().numpy()])
            test_predictions.extend([(torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
        sys.stdout.write('\n')
    test_labels = np.array(test_labels)
    test_labels = test_labels.reshape(len(test_labels), -1)
    test_predictions = np.array(test_predictions)
    _, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes)
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    y_pred, y_true = inverse_convert_label(test_predictions), inverse_convert_label(test_labels)
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    avg = np.mean([p, r, acc])
    return p, r, acc, avg


def multi_label_roc(labels, predictions, num_classes):
    thresholds, thresholds_optimal, aucs = [], [], []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    if len(labels.shape) == 1:
        labels = labels[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def main():
    parser = argparse.ArgumentParser(description='Train MIL Models with ReMix')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0, ), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='Camelyon16', type=str,
                        choices=['Camelyon16', 'Unitopatho'], help='Dataset folder name')
    parser.add_argument('--model', default='dsmil', type=str,
                        choices=['dsmil', 'abmil'], help='MIL model')
    # ReMix Parameters
    parser.add_argument('--num_prototypes', default=None, type=int, help='Number of prototypes per bag')
    parser.add_argument('--mode', default=None, type=str,
                        choices=['None', 'replace', 'append', 'interpolate', 'cov', 'joint'],
                        help='Augmentation method')
    parser.add_argument('--rate', default=0.5, type=float, help='Augmentation rate')
    
    # Utils
    parser.add_argument('--exp_name', required=True, help='exp_name')
    parser.add_argument('--data_root', required=False, default='datasets', type=str, help='path to data root')
    parser.add_argument('--num_repeats', default=10, type=int, help='Number of repeats')
    args = parser.parse_args()
    
    assert args.dataset in ['Camelyon16', 'Unitopatho'], 'Dataset not supported'
    # For Camelyon16, we follow DSMIL to use binary labels: 1 for positive bags and 0 for negative bags.
    # For Unitopatho, we use one-hot encoding.
    args.num_classes = {'Camelyon16': 1, 'Unitopatho': 6}[args.dataset]
    train_labels_pth = f'{args.data_root}/{args.dataset}/remix_processed/train_bag_labels.npy'
    test_labels_pth = f'{args.data_root}/{args.dataset}/remix_processed/test_bag_labels.npy'

    # loading the list of test data
    test_feats = open(f'{args.data_root}/{args.dataset}/remix_processed/test_list.txt', 'r').readlines()
    test_feats = np.array(test_feats)
    
    # use first_time to avoid duplicated logs
    first_time = True
    for t in range(args.num_repeats):
        ckpt_pth = setup_logger(args, first_time)
        logging.info(f'current args: {args}')
        logging.info(f'augmentation mode: {args.mode}')

        # prepare model
        if args.model == 'abmil':
            milnet = abmil.BClassifier(args.feats_size, args.num_classes).cuda()
        elif args.model == 'dsmil':
            i_classifier = dsmil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
            b_classifier = dsmil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=0).cuda()
            milnet = dsmil.MILNet(i_classifier, b_classifier).cuda()
            state_dict_weights = torch.load('init.pth')
            milnet.load_state_dict(state_dict_weights, strict=False)
            logging.info('loading from init.pth')

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

        if args.num_prototypes is not None:
            # load reduced-bag
            train_feats_pth = f'{args.data_root}/{args.dataset}/remix_processed/train_bag_feats_proto_{args.num_prototypes}.npy'
            logging.info(f'loading train_feats from {train_feats_pth}')
            # loading features
            train_feats = np.load(train_feats_pth, allow_pickle=True)
            train_feats = torch.Tensor(train_feats).cuda()

            if args.mode == 'cov' or args.mode == 'joint':
                # loading semantic shift vectors
                train_shift_bank_pth = f'{args.data_root}/{args.dataset}/remix_processed/train_bag_feats_shift_{args.num_prototypes}.npy'
                semantic_shifts = np.load(f'{train_shift_bank_pth}')
            else:
                semantic_shifts = None
        else:
            # when train_feats is None, loading them directly from the dataset npy folder.
            train_feats = open(f'{args.data_root}/{args.dataset}/remix_processed/train_list.txt', 'r').readlines()
            train_feats = np.array(train_feats)
            semantic_shifts = None
            
        # loading labels
        train_labels, test_labels = np.load(train_labels_pth), np.load(test_labels_pth)
        train_labels, test_labels = torch.Tensor(train_labels).cuda(), torch.Tensor(test_labels).cuda()
        
        for epoch in range(1, args.num_epochs + 1):
            # shuffle data
            shuffled_train_idxs = np.random.permutation(len(train_labels))
            train_feats, train_labels = train_feats[shuffled_train_idxs], train_labels[shuffled_train_idxs]
            train_loss_bag = train(train_feats, train_labels, milnet, criterion, optimizer, args, semantic_shifts)
            logging.info('Epoch [%d/%d] train loss: %.4f' % (epoch, args.num_epochs, train_loss_bag))
            scheduler.step()

        precision, recall, accuracy, avg = test(test_feats, test_labels, milnet, criterion, args)
        torch.save(milnet.state_dict(), ckpt_pth)
        logging.info('Final model saved at: ' + ckpt_pth)
        logging.info(f'Precision, Recall, Accuracy, Avg')
        logging.info(f'{precision*100:.2f} {recall*100:.2f} {accuracy*100:.2f} {avg*100:.2f}')
        first_time = False


if __name__ == '__main__':
    main()
