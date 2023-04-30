import torch
from operator import itemgetter
from copy import deepcopy
import pickle

class Confident_Sample():
    # confident all samples, confident new samples.
    # confident new samples idx
    # confident all mask, confident new samples mask (for unlabeled dataset only, not applicable to MergeDataset).

    def __init__(self, args):
        self.step = args.step
        self.warmup = args.warmup
        self.dist = args.dist
        self.C_star_previous = args.C_star_previous
        self.num_C = args.num_unlabeled_classes_predicted
        self.device = 'cuda:0'
        self.conf_thres = args.conf_thres
        self.mean_var_dir = args.mean_var_dir
        self.saved_whole_prev = []
        self.saved_whole = []
        self.model_path = args.model_path

    def get_mask_confident(self, uc_centers, new_feats, new_preds):
        idx_confident = []
        pdist = torch.nn.PairwiseDistance(p=2)
        indices = torch.range(0, new_preds.size(dim=0) - 1).to(self.device)
        for idx, cluster_center in enumerate(uc_centers):
            feats = new_feats[new_preds == idx]
            idx_indices = indices[new_preds == idx]
            N = round(self.conf_thres * idx_indices.size(dim=0))
            dist = pdist(cluster_center, feats)
            conf_idx = torch.argsort(dist)[:N]
            idx_confident.extend([idx_indices[j] for j in conf_idx])

        mask = [False] * new_preds.size(dim=0)
        for i in idx_confident:
            mask[i.to(torch.int32)] = True

        # mask = [True if i in idx_confident else False for i in range(new_preds.size(dim=0))]
        # print(1)

        return mask

    def get_conf_new_idx(self, uc_centers, uc_preds, feats, idx_old, uc_idx):
        self.idx_new = [i for i in range(len(uc_centers)) if i not in self.idx_old]
        self.mask_new = [True if i in self.idx_new else False for i in uc_preds]
        self.mask_old = [False if i in self.idx_new else True for i in uc_preds]
        new_feats = feats[self.mask_new]
        new_preds = uc_preds[self.mask_new]

        self.mask_conf_new = self.get_mask_confident(uc_centers,new_feats,new_preds)

        uc_idx_conf_new = uc_idx[self.mask_new]
        uc_idx_conf_new = uc_idx_conf_new[self.mask_conf_new]

        conf_new_feats = new_feats[self.mask_conf_new]
        conf_new_preds = new_preds[self.mask_conf_new]

        self.uc_idx_dict_new = self.get_UC_idx_dic(uc_idx_conf_new, conf_new_preds)

        self.get_conf_old_idx(uc_centers, uc_preds, feats,uc_idx)

        return conf_new_feats, conf_new_preds


    def get_conf_all_idx(self, uc_centers, uc_preds, feats, uc_idx):
        self.mask_conf_all = self.get_mask_confident(uc_centers, feats, uc_preds)

        # args.conf_thres = args.conf_thres + 0.1
        uc_idx = uc_idx[self.mask_conf_all]
        self.uc_conf_preds = uc_preds[self.mask_conf_all]

        uc_idx_dict = self.get_UC_idx_dic(uc_idx, self.uc_conf_preds)

        self.uc_idx_dict_all = uc_idx_dict

    def get_conf_old_idx(self, uc_centers, uc_preds, feats, uc_idx):
        old_feats = feats[self.mask_old]
        old_preds = uc_preds[self.mask_old]
        self.mask_conf_old = self.get_mask_confident(uc_centers, old_feats, old_preds)

        uc_idx_conf_old = uc_idx[self.mask_old]
        uc_idx_conf_old = uc_idx_conf_old[self.mask_conf_old]

        conf_old_preds = old_preds[self.mask_conf_old]
        for i in range(len(conf_old_preds)):
            conf_old_preds[i] = self.old_idx_map[conf_old_preds[i].item()]

        self.uc_idx_dict_old = self.get_UC_idx_dic(uc_idx_conf_old, conf_old_preds)


    def identify_old_clusters(self, uc_centers, labeled_mean):
        candidates = []
        idx_old = []
        C_old = []
        old_idx_map = {}
        pdist = torch.nn.PairwiseDistance(p=2)
        for idx, cluster in enumerate(uc_centers):
            for j in labeled_mean.keys():
                dist_labeled_unlabeled = pdist(cluster, labeled_mean[j])
                if dist_labeled_unlabeled < self.dist:
                    candidates.append([j, idx, dist_labeled_unlabeled])

        # Find old clusters
        candidates = sorted(candidates, key=itemgetter(2))
        print('candidates', candidates[:10])
        for candidate in candidates:
            if candidate[0] not in C_old and candidate[1] not in idx_old and len(C_old) <= len(self.C_star_previous):
                C_old.append(candidate[0])
                idx_old.append(candidate[1])
                old_idx_map[candidate[1]] = candidate[0]
        print('number of old clusters identified', len(C_old))
        print(C_old)

        self.old_idx_map = old_idx_map
        self.idx_old = idx_old
        self.C_old = C_old
        self.new_st = max(self.C_star_previous) + 1  # Start new label with max old label + 1
        self.C_new = [(self.new_st + i) for i in range(self.num_C - len(self.C_old))]
        self.C_star = self.C_star_previous + self.C_new

        return self.idx_old

    def identify_old_clusters_incre(self, uc_centers, saved_whole_prev):
        self.saved_whole_prev = deepcopy(saved_whole_prev)
        pdist = torch.nn.PairwiseDistance(p=2)
        idx_old = []
        C_old = []  # re identify old clusters
        candidates = []
        old_idx_map = {}
        for idx, cluster in enumerate(uc_centers):
            for mean_list in self.saved_whole_prev:
                for mean in mean_list:
                    dist_old_vs_unlabeled = pdist(cluster, mean[0])
                    if dist_old_vs_unlabeled < self.dist:
                        candidates.append([mean[3], idx, dist_old_vs_unlabeled])

        candidates = sorted(candidates, key=itemgetter(2))
        print('candidates', candidates)
        for candidate in candidates:
            if candidate[0] not in C_old and candidate[1] not in idx_old and len(C_old) <= len(self.C_star_previous):
                C_old.append(candidate[0])
                idx_old.append(candidate[1])
                old_idx_map[candidate[1]] = candidate[0]
        print('number of old clusters identified', len(C_old))
        print(C_old)

        self.old_idx_map = old_idx_map
        self.idx_old = idx_old
        self.C_old = C_old
        self.new_st = max(self.C_star_previous) + 1  # Start new label with max old label + 1
        self.C_new = [(self.new_st + i) for i in range(self.num_C - len(self.C_old))]
        self.C_star = self.C_star_previous + self.C_new



    def get_UC_idx_dic(self, cluster_idx, preds):
        dic = {}
        for idx, i in enumerate(cluster_idx):
            dic[i.item()] = preds[idx]

        return dic

    def save_means(self, labeled_mean, labeled_variances, labeled_class_N, uc_centers, uc_preds, feats, idx_old, uc_idx):
        conf_new_feats, conf_new_preds = self.get_conf_new_idx(uc_centers, uc_preds, feats, idx_old, uc_idx)

        saved_old_clusters = []
        saved_new_clusters = []
        for idx, i in enumerate(list(set(conf_new_preds.tolist()))):
            this_feats = conf_new_feats[conf_new_preds == i]
            mean_feats = this_feats.mean(dim=0)
            cov_feats = torch.transpose(this_feats, 0, 1).cov()
            cov_feats += torch.eye(cov_feats.size(dim=0)).to('cuda:0') * torch.diagonal(cov_feats,0).min()
            class_N = this_feats.size(dim=0)
            saved_new_clusters.append(
                [mean_feats, cov_feats, class_N, self.C_new[idx]])  # Assign new labels to new clusters

        # Update ?
        for i in labeled_mean.keys():
            saved_old_clusters.append([labeled_mean[i], labeled_variances[i], labeled_class_N[i], i])

        self.saved_whole = [saved_old_clusters, saved_new_clusters]

        save_path = self.model_path[:-3]+f'_mean{self.step}.pt'
        with open(save_path, 'wb') as f:
            pickle.dump(self.saved_whole, f)
        print(f'step {self.step} meanlist saved to', save_path)

    def save_means_incre(self, uc_centers, uc_preds, feats, uc_idx):
        conf_new_feats, conf_new_preds = self.get_conf_new_idx(uc_centers, uc_preds, feats, self.idx_old, uc_idx)
        saved_new_clusters = []

        # Note here, use statistics of CONF feats
        for idx, i in enumerate(list(set(conf_new_preds.tolist()))):
            this_feats = conf_new_feats[conf_new_preds == i]
            mean_feats = this_feats.mean(dim=0)
            cov_feats = torch.transpose(this_feats, 0, 1).cov()
            cov_feats += torch.eye(cov_feats.size(dim=0)).to('cuda:0') * torch.diagonal(cov_feats,0).min()
            class_N = this_feats.size(dim=0)
            saved_new_clusters.append([mean_feats, cov_feats, class_N, self.C_new[idx]])

        saved_old_clusters = []
        old_feats, old_preds = (feats[self.mask_old])[self.mask_conf_old], (uc_preds[self.mask_old])[self.mask_conf_old]
        for idx, i in enumerate(list(set(old_preds.tolist()))):
            this_feats = old_feats[old_preds == i]
            mean_feats = this_feats.mean(dim=0)
            cov_feats = torch.transpose(this_feats, 0, 1).cov()
            cov_feats += torch.eye(cov_feats.size(dim=0)).to('cuda:0') * torch.diagonal(cov_feats,0).min()
            class_N = this_feats.size(dim=0)
            saved_old_clusters.append([mean_feats, cov_feats, class_N, self.old_idx_map[i]])

        svd_whole = self.update_means_incre(self.saved_whole_prev, saved_old_clusters, saved_new_clusters)
        self.saved_whole.append(svd_whole)

        save_path = self.model_path[:-3]+f'_mean{self.step}.pt'
        with open(save_path, 'wb') as f:
            pickle.dump(self.saved_whole, f)
        print(f'step {self.step} meanlist saved to', save_path)


    def update_means_incre(self, whole_prev, old, new):
        saved_whole = []
        temp = deepcopy(whole_prev)
        for i, M in enumerate(whole_prev):
            FM = 0
            for j, (X, Q, n, c) in enumerate(M):
                for X_, Q_, n_, c_ in old:
                    if c == c_:
                        X_new, Q_new, n_new = self.merge_means(X,X_,Q,Q_,n,n_)
                        temp[i][j] = [X_new,Q_new,n_new,c]
            if FM == len(M):
                temp[i] += new
                saved_whole = deepcopy(temp)
                return saved_whole

        saved_whole = deepcopy(temp.append([new]))

        return saved_whole


    def merge_means(self, x1, x2, Q1, Q2, n1, n2):
        n = n1+n2
        x = (n1*x1 + n2*x2) / n
        Q = (n1-1)*Q1 + (n2-1)*Q2 + n1*torch.matmul((x1-x).view(x1.size(0),1), (x1-x).view(1, x1.size(0))) + n2*torch.matmul((x2-x).view(x1.size(0),1), (x2-x).view(1, x1.size(0)))
        Q = Q / (n-1)

        return x, Q, n
