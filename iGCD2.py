import argparse
import os
import pickle
import random
from operator import itemgetter
from copy import deepcopy

from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from project_utils.cluster_utils import mixed_eval, AverageMeter
from models import vision_transformer as vits

from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights, PairEnum
from conf_sample import Confident_Sample
from project_utils.cluster_utils import cluster_acc

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits, get_incre_datasets

from tqdm import tqdm

from torch.nn import functional as F

from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path
import wandb

# TODO: Debug
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# os.chdir('/userhome/cs/yihuac/finalproject/generalized-category-discovery/')

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def info_nce_logits(features, args):
    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels


def get_conf_sup(features, uq_idxs, uc_idx_dic, mask_lab, class_labels, conf_acc_record, old=False):
    conf_labels = torch.Tensor([]).to(device)
    true_labels = torch.Tensor([]).to(device)
    sup_conf_feats1 = torch.Tensor([]).to(device)
    sup_conf_feats2 = torch.Tensor([]).to(device)
    f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
    class_lab_ = class_labels[~mask_lab]
    un_uq_idx = uq_idxs[~mask_lab]
    for idx, i in enumerate(un_uq_idx.tolist()):
        if i in uc_idx_dic.keys():
            conf_labels = torch.cat([conf_labels, uc_idx_dic[i].reshape(1)],
                                    dim=0)
            sup_conf_feats1 = torch.cat([sup_conf_feats1, f1[idx].unsqueeze(0)], dim=0)
            sup_conf_feats2 = torch.cat([sup_conf_feats2, f2[idx].unsqueeze(0)], dim=0)
            true_labels = torch.cat([true_labels, class_lab_[idx].reshape(1)], dim=0)
    with torch.no_grad():
        if conf_labels.size(dim=0) != 0:
            if old:
                temp_acc = ((conf_labels == true_labels).float().sum() / conf_labels.size(dim=0)).item()
                conf_acc_record.update(temp_acc, conf_labels.size(dim=0))
            else:
                conf_acc_record.update(cluster_acc(true_labels.to(torch.int).detach().cpu().numpy(), conf_labels.to(torch.int).detach().cpu().numpy()), conf_labels.size(dim=0))
    sup_conf_feats = torch.cat([sup_conf_feats1.unsqueeze(1), sup_conf_feats2.unsqueeze(1)],
                               dim=1)  # [labeled samples * 2, 65536]

    return sup_conf_feats, conf_labels


def train_incremental_w_labeled(projection_head, model, train_loader, test_loader, unlabelled_train_loader, args,
                                train_loader_labeled=None, saved_whole_prev=None):
    optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )

    sup_con_crit = SupConLoss()
    best_test_acc_lab = 0
    best_all_acc = 0.0
    # args.model_path = './testing_cifar10.pt'

    uc_centers = []
    args.C_star_previous = list(args.labeled_classes)  # set of seen classes at previous step
    args.C_test = list(set(args.C_step0 + args.C_step1))
    uc_preds = []

    conf_all = Confident_Sample(args)

    for epoch in range(args.epochs):

        if epoch == args.warmup:
            args.sup_con_weight = 0.25

        loss_record = AverageMeter()
        train_acc_record = AverageMeter()
        conf_acc_record = AverageMeter()

        projection_head.train()
        model.train()

        for batch_idx, batch in enumerate(tqdm(train_loader)):

            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
            images = torch.cat(images, dim=0).to(
                device)  # 2 Views [[bs, 3, 224, 224], [bs, 3, 224, 224]] --> [2*bs, 3, 224, 224]

            # Extract features with base model
            features_a = model(images)  # [bs, 768]

            # Pass features through projection head
            features = projection_head(features_a)  # [bs, 65536]

            # L2-normalize features
            features = torch.nn.functional.normalize(features, dim=-1)

            # Choose which instances to run the contrastive loss on
            if args.contrast_unlabel_only:
                # Contrastive loss only on unlabelled instances
                f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                con_feats = torch.cat([f1, f2], dim=0)  # [unlabeled samples * 2, 65536]
            else:
                # Contrastive loss for all examples
                con_feats = features

            contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats, args=args)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # Supervised contrastive loss
            f1, f2 = [f[mask_lab] for f in features.chunk(2)]
            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # [labeled samples * 2, 65536]
            sup_con_labels = class_labels[mask_lab]  # [labeled samples]

            if epoch >= args.warmup and args.conf_new_supcon and args.conf == 'split':
                old_conf_feats, old_conf_labels = get_conf_sup(features, uq_idxs, conf_all.uc_idx_dict_old, mask_lab, class_labels, conf_acc_record, old=True)
                if old_conf_labels.size(dim=0) != 0:
                    sup_con_feats = torch.cat([sup_con_feats, old_conf_feats], dim=0)
                    sup_con_labels = torch.cat([sup_con_labels, old_conf_labels], dim=0)

            sup_con_loss = 0.0
            if sup_con_labels.size(dim=0) != 0:
                sup_con_loss += sup_con_crit(sup_con_feats, labels=sup_con_labels)

            # Pseudo-label supervised contrastive learning
            pl_loss = 0.0
            if args.PL_supcon:
                rank_feat, rank_feat_bar = [torch.nn.functional.normalize(f, dim=-1).detach() for f in features_a.chunk(2)]
                if args.coss_sup:
                    feat_row, feat_col = PairEnum(rank_feat)
                    tmp_distance_ori = torch.bmm(feat_row.view(feat_row.size(0), 1, -1), feat_col.view(feat_row.size(0), -1, 1))
                    tmp_distance_ori = tmp_distance_ori.squeeze()
                    target_ulb = torch.zeros_like(tmp_distance_ori).float()
                    target_ulb[tmp_distance_ori > args.costhre] = 1
                    target_ulb = target_ulb.view(rank_feat.size(0), rank_feat.size(0))

                elif args.rankst_sup:
                    rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
                    rank_idx1, rank_idx2 = PairEnum(rank_idx)
                    rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
                    rank_idx1, _ = torch.sort(rank_idx1, dim=1)
                    rank_idx2, _ = torch.sort(rank_idx2, dim=1)
                    rank_diff = rank_idx1 - rank_idx2
                    rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
                    target_ulb = torch.ones_like(rank_diff).float().to(device)
                    target_ulb[rank_diff > 0] = 0
                    target_ulb = target_ulb.view(rank_feat.size(0), rank_feat.size(0))
                rank_feat, rank_feat_bar = [f for f in features.chunk(2)]
                sup_con_feats = torch.cat([rank_feat.unsqueeze(1), rank_feat_bar.unsqueeze(1)], dim=1)
                pl_loss += sup_con_crit(sup_con_feats, mask=target_ulb)


            # sup contrastive loss for confident samples of new classes
            if epoch >= args.warmup and args.conf_new_supcon:
                dict = None
                if args.conf == 'new' or args.conf == 'split':
                    dict = conf_all.uc_idx_dict_new
                elif args.conf == 'all':
                    dict = conf_all.uc_idx_dict_all
                sup_conf_feats, conf_labels = get_conf_sup(features, uq_idxs, dict, mask_lab, class_labels, conf_acc_record)
                if conf_labels.size(dim=0) != 0:
                    sup_con_loss += args.lam_pl * sup_con_crit(sup_conf_feats, labels=conf_labels)

            # Total loss
            loss = (1 - args.sup_con_weight) * contrastive_loss + args.sup_con_weight * sup_con_loss + args.lam_pl * pl_loss

            # Train acc
            _, pred = contrastive_logits.max(1)
            acc = (pred == contrastive_labels).float().mean().item()
            train_acc_record.update(acc, pred.size(0))

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch, loss_record.avg,
                                                                                  train_acc_record.avg))
        if epoch % args.warmup == 0 and epoch != 0 and args.conf_new_supcon:

            print('conf cluster acc:', conf_acc_record.avg)

        with torch.no_grad():

            print('Testing on unlabelled examples in the training data...')
            if (epoch + 1) % args.warmup == 0:
                all_acc, old_acc, new_acc, uc_centers, uc_preds, unlabeled_feats, uc_idx = test_kmeans(
                    model, unlabelled_train_loader,
                    epoch=epoch,
                    save_name='Train ACC Unlabelled',
                    args=args, predsfeats=True, test=False)
                unlabeled_feats = torch.Tensor(unlabeled_feats).to(device)
                uc_preds = torch.Tensor(uc_preds).to(torch.int).to(device)
                uc_idx = torch.Tensor(uc_idx).to(torch.int).to(device)
                uc_centers = torch.Tensor(uc_centers).to(device)
            else:
                all_acc, old_acc, new_acc, uc_centers = test_kmeans(model, unlabelled_train_loader,
                                                                                   epoch=epoch,
                                                                                   save_name='Train ACC Unlabelled',
                                                                                   args=args, test=False)

            print('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test, cluster_centers_test = test_kmeans(model, test_loader,
                                                                                         epoch=epoch,
                                                                                         save_name='Test ACC',
                                                                                         args=args, test=True)

        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                              new_acc))
        print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test,
                                                                             new_acc_test))
        wandb.log({'ACC_all':all_acc, 'ACC_old':old_acc, 'ACC_new':new_acc, 'ACC_all_test':all_acc_test, 'ACC_old_test':old_acc_test, 'ACC_new_test':new_acc_test, 'conf cluster acc:': conf_acc_record.avg}, step=epoch)
        # Step schedule
        exp_lr_scheduler.step()

        torch.save(model.state_dict(), args.model_path)
        print("model saved to {}.".format(args.model_path))

        torch.save(projection_head.state_dict(), args.model_path[:-3] + '_proj_head.pt')
        print("projection head saved to {}.".format(args.model_path[:-3] + '_proj_head.pt'))



        with torch.no_grad():
            if (epoch + 1) % args.warmup == 0 and args.step == 1:
                labeled_mean, labeled_cov, labeled_class_N = generate_mean_var(model, train_loader_labeled, args)
                idx_old = conf_all.identify_old_clusters(uc_centers, labeled_mean)
                conf_all.get_conf_all_idx(uc_centers, uc_preds, unlabeled_feats, uc_idx)
                conf_all.save_means(labeled_mean, labeled_cov, labeled_class_N, uc_centers, uc_preds, unlabeled_feats, idx_old, uc_idx)

            if all_acc > best_all_acc:
                print(f'Best overall ACC on the unlabeled data set: {all_acc:.4f}...')
                print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                           new_acc))

                torch.save(model.state_dict(), args.model_path[:-3] + f'_best.pt')
                print("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

                torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_best.pt')
                print("projection head saved to {}.".format(args.model_path[:-3] + f'_proj_head_best.pt'))

                best_all_acc = all_acc


def train_incremental_wo_labeled(projection_head, model, old_model, train_loader, test_loader, unlabelled_train_loader,
                                 args):
    optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )

    sup_con_crit = SupConLoss()
    best_test_acc_lab = 0
    best_all_acc = 0.0

    args.C_test = list(set(args.C_step0 + args.C_step1 + args.C_step2))
    uc_preds = []

    conf_all = Confident_Sample(args)

    for epoch in range(args.epochs):

        loss_record = AverageMeter()
        train_acc_record = AverageMeter()
        conf_acc_record = AverageMeter()

        projection_head.train()
        model.train()

        for batch_idx, batch in enumerate(tqdm(train_loader)):

            images, class_labels, uq_idxs = batch

            class_labels = class_labels.to(device)
            images = torch.cat(images, dim=0).to(
                device)  # 2 Views [[bs, 3, 224, 224], [bs, 3, 224, 224]] --> [2*bs, 3, 224, 224]

            # Extract features with base model
            features_a = model(images)  # [bs, 768]

            # Knowledge Distillation
            loss_kd = 0.0
            if args.kd:
                new_model_features = torch.nn.functional.normalize(features_a, dim=-1)
                with torch.no_grad():
                    old_model_features = old_model(images)
                    old_model_features = torch.nn.functional.normalize(old_model_features, dim=-1)
                loss_kd = torch.dist(new_model_features, old_model_features)

            sup_con_loss = 0.0
            # Pseudo-label supervised contrastive learning

            # Pass features through projection head
            features = projection_head(features_a)  # [bs, 65536]

            # L2-normalize features
            features = torch.nn.functional.normalize(features, dim=-1)

            # Choose which instances to run the contrastive loss on
            con_feats = features
            contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats, args=args)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            pl_loss = 0.0
            if args.PL_supcon:
                rank_feat, rank_feat_bar = [torch.nn.functional.normalize(f, dim=-1).detach() for f in features_a.chunk(2)]
                if args.coss_sup:
                    feat_row, feat_col = PairEnum(rank_feat)
                    tmp_distance_ori = torch.bmm(feat_row.view(feat_row.size(0), 1, -1), feat_col.view(feat_row.size(0), -1, 1))
                    tmp_distance_ori = tmp_distance_ori.squeeze()
                    target_ulb = torch.zeros_like(tmp_distance_ori).float()
                    target_ulb[tmp_distance_ori > args.costhre] = 1
                    target_ulb = target_ulb.view(rank_feat.size(0), rank_feat.size(0))

                elif args.rankst_sup:
                    rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
                    rank_idx1, rank_idx2 = PairEnum(rank_idx)
                    rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
                    rank_idx1, _ = torch.sort(rank_idx1, dim=1)
                    rank_idx2, _ = torch.sort(rank_idx2, dim=1)
                    rank_diff = rank_idx1 - rank_idx2
                    rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
                    target_ulb = torch.ones_like(rank_diff).float().to(device)
                    target_ulb[rank_diff > 0] = 0
                    target_ulb = target_ulb.view(rank_feat.size(0), rank_feat.size(0))
                rank_feat, rank_feat_bar = [f for f in features.chunk(2)]
                sup_con_feats = torch.cat([rank_feat.unsqueeze(1), rank_feat_bar.unsqueeze(1)], dim=1)
                pl_loss += sup_con_crit(sup_con_feats, mask=target_ulb)

            fr_loss = 0.0
            if args.feature_replay:
                feats_list, label_list = sample_feats(args.saved_whole_prev)
                for idx, feats in enumerate(feats_list):

                    label_ = label_list[idx]
                    feats = projection_head(feats)
                    feats = torch.nn.functional.normalize(feats, dim=-1)
                    f1, f2 = [f for f in feats.chunk(2)]
                    sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                    fr_loss += sup_con_crit(sup_con_feats, labels=label_)

            # sup contrastive loss for confident samples of new classes
            if epoch >= args.warmup and args.conf_new_supcon:
                dict = None
                if args.conf == 'new' or args.conf == 'split':
                    dict = conf_all.uc_idx_dict_new
                elif args.conf == 'all':
                    dict = conf_all.uc_idx_dict_all
                sup_conf_feats, conf_labels = get_conf_sup(features, uq_idxs, dict, [False]*class_labels.size(dim=0), class_labels, conf_acc_record)
                if conf_labels.size(dim=0) != 0:
                    sup_con_loss += sup_con_crit(sup_conf_feats, labels=conf_labels)

            # Total loss
            loss = (1 - args.sup_con_weight) * contrastive_loss + args.lam_pl * pl_loss + args.lam_fr * fr_loss + args.kd_weight * loss_kd

            # Train acc
            _, pred = contrastive_logits.max(1)
            acc = (pred == contrastive_labels).float().mean().item()
            train_acc_record.update(acc, pred.size(0))

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch, loss_record.avg,
                                                                                  train_acc_record.avg))
        if epoch % args.warmup == 0 and epoch != 0 and args.conf_new_supcon:

            print('conf cluster acc:', conf_acc_record.avg)

        with torch.no_grad():

            print('Testing on unlabelled examples in the training data...')
            if (epoch + 1) % args.warmup == 0:
                all_acc, old_acc, new_acc, uc_centers, uc_preds, u_feats , uc_idx = test_kmeans(
                    model, unlabelled_train_loader,
                    epoch=epoch,
                    save_name='Train ACC Unlabelled',
                    args=args, predsfeats=True, test=False)
                u_feats = torch.Tensor(u_feats).to(device)
                uc_preds = torch.Tensor(uc_preds).to(torch.int).to(device)
                uc_idx = torch.Tensor(uc_idx).to(torch.int).to(device)
                uc_centers = torch.Tensor(uc_centers).to(device)
            else:
                all_acc, old_acc, new_acc, uc_centers = test_kmeans(model, unlabelled_train_loader,
                                                                                   epoch=epoch,
                                                                                   save_name='Train ACC Unlabelled',
                                                                                   args=args, test=False)

            print('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test, cluster_centers_test = test_kmeans(model, test_loader,
                                                                                         epoch=epoch,
                                                                                         save_name='Test ACC',
                                                                                         args=args, test=True)

        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                              new_acc))
        print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test,
                                                                             new_acc_test))

        wandb.log({'ACC_all':all_acc, 'ACC_old':old_acc, 'ACC_new':new_acc, 'ACC_all_test':all_acc_test, 'ACC_old_test':old_acc_test, 'ACC_new_test':new_acc_test, 'conf cluster acc:': conf_acc_record.avg}, step=epoch)
        # Step schedule
        exp_lr_scheduler.step()

        torch.save(model.state_dict(), args.model_path)
        print("model saved to {}.".format(args.model_path))

        torch.save(projection_head.state_dict(), args.model_path[:-3] + '_proj_head.pt')
        print("projection head saved to {}.".format(args.model_path[:-3] + '_proj_head.pt'))

        with torch.no_grad():
            if (epoch + 1) % args.warmup == 0 and args.step > 1:
                assert args.saved_whole_prev is not None
                conf_all.identify_old_clusters_incre(uc_centers, args.saved_whole_prev)

                # conf_all.get_conf_all_idx(uc_centers, uc_preds, u_feats, uc_idx)
                # conf_all.save_means_incre(uc_centers, uc_preds, u_feats, uc_idx)


            if all_acc > best_all_acc:
                print(f'Best overall ACC on the unlabeled data set: {all_acc:.4f}...')
                print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                           new_acc))

                torch.save(model.state_dict(), args.model_path[:-3] + f'_best.pt')
                print("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

                torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_best.pt')
                print("projection head saved to {}.".format(args.model_path[:-3] + f'_proj_head_best.pt'))

                best_all_acc = all_acc


def get_mask_confident_new_unlabeled(uc_centers, unlabeled_new_feats, unlabeled_new_preds):
    idx_confident = []
    pdist = torch.nn.PairwiseDistance(p=2)
    indices = torch.range(0, unlabeled_new_preds.size(dim=0) - 1).to(device)
    for idx, cluster_center in enumerate(uc_centers):
        feats = unlabeled_new_feats[unlabeled_new_preds == idx]
        idx_indices = indices[unlabeled_new_preds == idx]
        N = round(0.2 * idx_indices.size(dim=0))
        dist = pdist(cluster_center, feats)
        conf_idx = torch.argsort(dist)[:N]
        idx_confident.extend([idx_indices[j] for j in conf_idx])

    mask = [False] * unlabeled_new_preds.size(dim=0)
    for i in idx_confident:
        mask[i.to(torch.int32)] = True

    # mask = [True if i in idx_confident else False for i in range(unlabeled_new_preds.size(dim=0))]
    # print(1)

    return mask

def conf_sample_acc(unlabeled_conf_idx, unlabeled_conf_ground_truth, unlabeled_cluster_uq_idx, unlabeled_cluster_preds):
    cluster_idx = unlabeled_cluster_uq_idx.tolist()
    t = np.array([])
    unlabeled_conf_ground_truth = unlabeled_conf_ground_truth.detach().cpu().numpy()
    unlabeled_cluster_preds = unlabeled_cluster_preds.to(torch.int).detach().cpu().numpy()

    for i in unlabeled_conf_idx:
        t = np.append(t, unlabeled_cluster_preds[cluster_idx.index(i)])

    return cluster_acc(unlabeled_conf_ground_truth, t.astype(int))

def sample_feats(saved_whole_prev):
    with torch.no_grad():
        num_pairs_per_class = 5
        feats_lists = []
        label_lists = []
        for mean_list in saved_whole_prev:
            this_feat1 = []
            this_feat2 = []
            this_label = []
            for mean in mean_list:
                cov = deepcopy(mean[1])
                cov += (torch.eye(cov.size(0)) * 1e-5).to(device)
                dist = torch.distributions.MultivariateNormal(mean[0], cov)
                this_feat1.extend(dist.sample((num_pairs_per_class,)).to(device))
                this_feat2.extend(dist.sample((num_pairs_per_class,)).to(device))
                this_label.extend(torch.ones(num_pairs_per_class).to(device) * mean[3])
            this_feat1.extend(this_feat2)
            feats_lists.append(torch.stack((this_feat1)))
            label_lists.append(torch.stack((this_label)))

        return feats_lists, label_lists


# def test_mean_var(projection_head, model, train_loader, test_loader, unlabelled_train_loader, args,
#                   train_loader_labeled):
#     print('Testing on unlabelled examples in the training data...')
#     model.eval()
#     with torch.no_grad():
#
#         all_acc, old_acc, new_acc, cluster_centers_unlabeled = test_kmeans(model, unlabelled_train_loader,
#                                                                            epoch=1, save_name='Train ACC Unlabelled',
#                                                                            args=args)
#
#         # TODO: if step = 1, save mean and variance for labeled data and confident unlabeled data.
#         class_mean, class_var = generate_mean_var(model, train_loader_labeled, args)
#         # TODO: get confident unlabeled data
#         # print(class_mean)
#         pdist = torch.nn.PairwiseDistance(p=2)
#         cos = torch.nn.CosineSimilarity(dim=0)
#         for i in class_mean.keys():
#             for idx, j in enumerate(cluster_centers_unlabeled):
#                 dist = pdist(class_mean[i].to(device), torch.from_numpy(j).to(device))
#                 cs = cos(class_mean[i].to(device), torch.from_numpy(j).to(device))
#                 print(f'for label {i} with {idx}-th cluster, dist is {dist}, cs is {cs}')


def generate_mean_var(model, train_loader_labeled, args):
    all_feat = []
    all_labels = []
    # class_mean = torch.zeros(args.num_labeled_classes, 768).cuda()
    # class_var = torch.zeros(args.num_labeled_classes, 768).cuda()

    class_mean = {}
    class_cov = {}
    class_N = {}
    model.eval()
    print('Extract Labeled Feature')
    for batch_idx, (x, label, uq_idx) in enumerate(tqdm(train_loader_labeled)):
        label = label.to(device)
        x = torch.cat(x, dim=0).to(device)
        feat = model(x)
        feat = torch.nn.functional.normalize(feat, dim=-1)

        all_feat.append(feat.detach().clone().cuda())
        all_labels.append(label.detach().clone().cuda())
        all_labels.append(label.detach().clone().cuda())

    # print(all_feat)
    # print(all_labels)

    all_feat = torch.cat(all_feat, dim=0).cuda()
    all_labels = torch.cat(all_labels, dim=0).cuda()
    #
    # print(all_feat.shape)
    # print(all_labels.shape)

    print('Calculate Labeled Mean-Var')
    for i in args.labeled_classes:
        # print(i)
        # print(all_labels == i)
        this_feat = all_feat[all_labels == i]
        this_mean = this_feat.mean(dim=0)
        this_var = torch.transpose(this_feat, 0, 1).cov()
        this_var += torch.eye(this_var.size(dim=0)).to(device) * torch.diagonal(this_var, 0).min()
        # class_mean[i, :] = this_mean
        # class_var[i, :] = (this_var + 1e-5)
        class_mean[i] = this_mean
        class_cov[i] = this_var
        class_N[i] = this_feat.size(dim=0)
    print('Finish')
    # class_mean, class_var = class_mean.cuda(), class_var.cuda()

    return class_mean, class_cov, class_N


def test_kmeans(model, test_loader,
                epoch, save_name,
                args, predsfeats=False, test=False):
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])
    whole_idx = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, uq_idx) in enumerate(tqdm(test_loader)):
        images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # mask = np.append(mask, np.array([True if x.item() in range(len(args.labeled_classes))
        #                                  else False for x in label]))
        mask = np.append(mask, np.array([True if x.item() in args.C_star_prev_gt
                                         else False for x in label]))
        whole_idx = np.append(whole_idx, uq_idx)

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_unlabeled_classes_predicted if test is False else len(args.C_test), random_state=0).fit(all_feats)
    preds = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    if predsfeats:
        return all_acc, old_acc, new_acc, cluster_centers, preds, all_feats, whole_idx
    else:
        return all_acc, old_acc, new_acc, cluster_centers


def check_dataset_classes(labeled_dataset, unlabeled_dataset, args):
    # Return the set of actual classes. It is found that the original method to load CUB dataset would result in 0 instances for some classes in the labeled data set, causing further issue of the model.
    # Hence, run this function to get the actual classes
    print('original labeled classes', len(args.train_classes), args.train_classes)
    print('original unlabeled old classes', len(args.train_classes), args.train_classes)
    print('original unlabeled new classes', len(args.unlabeled_classes), args.unlabeled_classes)
    if args.dataset_name == 'cub':
        all_labels = set(labeled_dataset.data.target)
    elif args.dataset_name == 'cifar10' or args.dataset_name == 'cifar100':
        all_labels = set(labeled_dataset.targets)
    labeled_classes = []
    for i in args.train_classes:
        if i in all_labels:
            labeled_classes.append(i)

    unlabeled_new_classes = list(args.unlabeled_classes)
    unlabeled_old_classes = []
    if args.dataset_name == 'cub':
        all_labels = set(unlabeled_dataset.data.target)
    elif args.dataset_name == 'cifar10' or args.dataset_name == 'cifar100':
        all_labels = set(unlabeled_dataset.targets)

    for i in args.train_classes:
        if i in all_labels:
            unlabeled_old_classes.append(i)

    temp = unlabeled_old_classes.copy()
    for i in unlabeled_old_classes:
        if i not in labeled_classes:
            # k = unlabeled_old_classes.pop(unlabeled_old_classes.index(i))
            unlabeled_new_classes.append(i)
            temp.remove(i)
    unlabeled_old_classes = temp.copy()

    print('updated labeled classes', len(labeled_classes), labeled_classes)
    print('updated unlabeled old classes', len(unlabeled_old_classes), unlabeled_old_classes)
    print('updated unlabeled new classes', len(unlabeled_new_classes), unlabeled_new_classes)

    args.labeled_classes = labeled_classes
    args.unlabeled_old_classes = unlabeled_old_classes
    args.unlabeled_new_classes = unlabeled_new_classes
    args.num_labeled_classes = len(labeled_classes)
    args.num_unlabeled_old_classes = len(unlabeled_old_classes)
    args.num_unlabeled_new_classes = len(unlabeled_new_classes)
    args.num_unlabeled_classes_predicted = args.num_unlabeled_old_classes + args.num_unlabeled_new_classes

    return args


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    parser = argparse.ArgumentParser(
        description='cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=False)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.5)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--conf_new_supcon', type=bool, default=True)
    parser.add_argument('--mini', type=bool, default=False)
    parser.add_argument('--mean_var_dir', type=str, default='./experiments/cifar10/mean_var_dir')

    parser.add_argument('--kd', type=bool, default=True)
    parser.add_argument('--kd_weight', type=float, default=0.1)
    parser.add_argument('--feature_replay',type=bool, default=True)
    parser.add_argument('--incre_exp', type=int, default=1)
    parser.add_argument('--dist', type=float, default=0.3)
    parser.add_argument('--PL_supcon', type=bool, default=False)
    parser.add_argument('--conf_thres',type=float, default=0.2)
    parser.add_argument('--wandb_project', type=str, default='exp')
    parser.add_argument('--conf', type=str, default='new')
    parser.add_argument('--lam_pl', type=float, default=0.2)
    parser.add_argument('--lam_fr',type=float, default=0.25)
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')

    args.dataset_name = 'cifar100'
    args.batch_size = 128
    args.grad_from_block = 11
    args.epochs = 200
    args.base_model = 'vit_dino'
    args.num_workers = 4
    args.use_ssb_splits = False
    args.sup_con_weight = 0.7
    args.weight_decay = 5e-5
    args.contrast_unlabel_only = False
    args.transform = 'imagenet'
    args.lr = 0.1
    args.eval_funcs = ['v1', 'v2']
    args.num_unlabeled_classes_predicted = 0
    args.warmup = 10
    args.conf_new_supcon = False
    args.mini = True
    args.step = 1
    args.incre_exp = 1
    args.dist = 0.3
    args.kd = False
    args.kd_weight = 0.1
    args.feature_replay = False
    args.PL_supcon = True
    args.conf_thres = 0.1
    args.coss_sup = True
    args.rankst_sup = False
    args.costhre = 0.9
    args.topk = 3
    args.conf = '~'
    args.lam_pl = 0.3
    args.lam_fr = 1

    args.old_model_path = '/home/a4/PROJECT/generalized-category-discovery/experiments/increment_test/model.pt'

    args.wandb_project = 'iGCD_cifar100_incre'
    wandb_run_name = f'{args.dataset_name} {args.step} incre test base'

    args = get_class_splits(args)


    # args.num_labeled_classes = len(args.train_classes)
    # args.num_unlabeled_classes = len(args.unlabeled_classes)  # num_unlabeled_classes is actually = num_new_classes
    # args.num_unlabeled_classes_predicted = args.num_labeled_classes + args.num_unlabeled_classes
    # print(args.num_labeled_classes, args.num_unlabeled_classes, args.num_unlabeled_classes_predicted)
    # print('old classes', args.train_classes)
    # print('new classes', args.unlabeled_classes)
    init_experiment(args, runner_name=['metric_learn_gcd'])
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    wandb.login(key='f60576576ab63ddac22116bae3b985acec394ac8')
    wandb.init(project=args.wandb_project,  # project='incd_dev_miu_improve_N_180-5-5-5-5',
               name=wandb_run_name,
               mode='online')


    print(args)
    args.model_path = '/home/a4/PROJECT/generalized-category-discovery/experiments/cifar100_increment_test/model.pt'
    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.base_model == 'vit_dino':

        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = dino_pretrain_path
        # pretrain_path =  '/home/a4/PROJECT/generalized-category-discovery/experiments/cifar10/metric_learn_gcd/log/(13.04.2023_|_11.852)/checkpoints/model.pt'
        # pretrain_path = '/home/a4/PROJECT/generalized-category-discovery/experiments/cifar10/metric_learn_gcd/log/(13.04.2023_|_32.734)/checkpoints/model.pt'
        model = vits.__dict__['vit_base']()
        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

        if args.warmup_model_dir is not None:
            print(f'Loading weights from {args.warmup_model_dir}')
            model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))


        old_model = None
        if args.step > 1:
            old_model = vits.__dict__['vit_base']()
            state_dict = torch.load(args.old_model_path, map_location='cpu')
            old_model.load_state_dict(state_dict)
            old_model.to(device)
            model.load_state_dict(state_dict)

        model.to(device)

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        # args.image_size = 40
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = 65536

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in model.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True

    else:

        raise NotImplementedError

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_incre_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    test_dataset = test_dataset[f'step{args.step}_test_dataset']

    # if args.step == 1:
    #     args = check_dataset_classes(train_dataset.labelled_dataset, train_dataset.unlabelled_dataset, args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    if args.step == 1:
        label_len = len(train_dataset.labelled_dataset)
        unlabelled_len = len(train_dataset.unlabelled_dataset)
        sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler if args.step == 1 else None, drop_last=True)
    train_loader_unlabeled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False)
    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                                out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    projection_head.to(device)

    # ----------------------
    # TRAIN
    # ----------------------


    # iGCD setting
    if args.step == 1:
        train_loader_labeled = DataLoader(dataset=train_dataset.labelled_dataset, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
        print(args.C_step0, args.C_step1)
        args.labeled_classes = args.C_step0
        args.num_unlabeled_classes_predicted = len(args.C_step1)
        args.C_star_prev_gt = list(set(args.C_step0))
        train_incremental_w_labeled(projection_head, model, train_loader, test_loader, train_loader_unlabeled,
                                    args, train_loader_labeled)

    elif args.step > 1:
        load_path = args.old_model_path[:-3]+f'_mean{args.step-1}.pt'
        with open(load_path, 'rb') as f:
            args.saved_whole_prev = pickle.load(f)
        print(args.C_step2)
        args.num_unlabeled_classes_predicted = len(args.C_step2)
        args.labeled_classes = []
        for mean_list in args.saved_whole_prev:
            for i in mean_list:
                args.labeled_classes.append(i[3])
        print(args.labeled_classes)
        args.C_star_previous = args.labeled_classes
        args.C_star_prev_gt = list(set(args.C_step0 + args.C_step1))
        train_incremental_wo_labeled(projection_head, model, old_model, train_loader, test_loader,
                                     train_loader_unlabeled,
                                     args)
