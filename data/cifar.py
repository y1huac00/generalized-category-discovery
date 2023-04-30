from torchvision.datasets import CIFAR10, CIFAR100
from copy import deepcopy
import numpy as np
from collections import Counter
from config import cifar_10_root, cifar_100_root
from random import shuffle

class CustomCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR10, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


class CustomCIFAR100(CIFAR100):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


def subsample_dataset(dataset, idxs):
    # Allow for setting in which all empty set of indices is passed

    if len(idxs) > 0:

        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]

        return dataset

    else:

        return None


def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):
    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):
    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def subsample_instances(dataset, prop_indices_to_subsample=0.8):
    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices


def get_cifar_10_datasets(train_transform, test_transform, train_classes=(0, 1, 8, 9),
                          prop_train_labels=0.8, split_train_val=False, seed=0, args=None):
    np.random.seed(seed)

    # Init entire training set
    original_whole_training_set = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)
    whole_training_set = None
    if args.mini:
        subsample_indices = subsample_instances(deepcopy(original_whole_training_set), prop_indices_to_subsample=0.1)
        whole_training_set = subsample_dataset(deepcopy(original_whole_training_set), subsample_indices)
    else:
        whole_training_set = original_whole_training_set

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = None
    if args.mini:
        train_dataset_unlabelled = subsample_dataset(deepcopy(original_whole_training_set),
                                                     np.array(list(unlabelled_indices)))
    else:
        train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)
    if args.mini:
        subsample_indices = subsample_instances(test_dataset, prop_indices_to_subsample=0.1)
        test_dataset = subsample_dataset(test_dataset, subsample_indices)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


def get_cifar_100_datasets(train_transform, test_transform, train_classes=range(80),
                           prop_train_labels=0.8, split_train_val=False, seed=0, args=None):
    np.random.seed(seed)

    # Init entire training set
    original_whole_training_set = CustomCIFAR100(root=cifar_100_root, transform=train_transform, train=True)
    whole_training_set = None
    if args.mini:
        subsample_indices = subsample_instances(deepcopy(original_whole_training_set), prop_indices_to_subsample=0.2)
        whole_training_set = subsample_dataset(deepcopy(original_whole_training_set), subsample_indices)
    else:
        whole_training_set = original_whole_training_set

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = None
    if args.mini:
        train_dataset_unlabelled = subsample_dataset(deepcopy(original_whole_training_set),
                                                     np.array(list(unlabelled_indices)))
    else:
        train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)
    if args.mini:
        subsample_indices = subsample_instances(test_dataset, prop_indices_to_subsample=0.2)
        test_dataset = subsample_dataset(test_dataset, subsample_indices)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


def incre_cifar10_exp1(train_transform=None, test_transform=None, args=None):
    class_dic = {0: [0.7, 0.15, 0.15], 1: [0.7, 0.15, 0.15], 2: [0.7, 0.3, 0],
                 3: [0.7, 0.3, 0], 4: [0.7, 0, 0.3], 5: [0, 1, 0],
                 6: [0, 1, 0], 7: [0, 0.7, 0.3], 8: [0, 0, 1], 9: [0, 0, 1]}

    np.random.seed(0)

    # Get who data set
    original_training_set = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)
    whole_training_set = deepcopy(original_training_set)
    if args.mini:
        subsample_indices = subsample_instances(deepcopy(whole_training_set), prop_indices_to_subsample=0.1)
        whole_training_set = subsample_dataset(deepcopy(whole_training_set), subsample_indices)

    whole_idx = whole_training_set.uq_idxs

    class_idx_dic = {}
    for i in class_dic.keys():
        this_split = class_dic[i]
        this_idx = [x for x, t in enumerate(whole_training_set.targets) if t == i]
        class_idx_dic[i] = np.split(this_idx, [int(len(this_idx) * this_split[0]),
                                               int(len(this_idx) * (this_split[0] + this_split[1]))])

    # get overall indx list for each dataset
    step0_idx, step1_idx, step2_idx = [], [], []
    for i in class_idx_dic.keys():
        step0_idx.extend(class_idx_dic[i][0])
        step1_idx.extend(class_idx_dic[i][1])
        step2_idx.extend(class_idx_dic[i][2])

    shuffle(step0_idx)
    shuffle(step1_idx)
    shuffle(step2_idx)

    step0_dataset = subsample_dataset(deepcopy(whole_training_set), step0_idx)
    step1_dataset = subsample_dataset(deepcopy(whole_training_set), step1_idx)
    step2_dataset = subsample_dataset(deepcopy(whole_training_set), step2_idx)

    test_dataset = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)
    if args.mini:
        subsample_indices = subsample_instances(test_dataset, prop_indices_to_subsample=0.1)
        test_dataset = subsample_dataset(test_dataset, subsample_indices)

    step1_test_dataset = subsample_classes(deepcopy(test_dataset), include_classes=[0, 1, 2, 3, 4, 5, 6, 7])

    all_datasets = {'step0_dataset': step0_dataset, 'step1_dataset': step1_dataset, 'step2_dataset': step2_dataset,
                    'test_dataset': {'step1_test_dataset':step1_test_dataset, 'step2_test_dataset':test_dataset}}

    return all_datasets

def incre_cifar100_exp1(train_transform=None, test_transform=None, args=None):
    class_dic = {}
    for i in range(100):
        if i < 10:
            class_dic[i] = [1, 0, 0]
        elif i < 20:
            class_dic[i] = [0.7, 0.3, 0]
        elif i < 80:
            class_dic[i] = [0.7, 0.15, 0.15]
        elif i < 90:
            class_dic[i] = [0, 0.7, 0.3]
        elif i < 100:
            class_dic[i] = [0, 0, 1]

    np.random.seed(0)

    # Get who data set
    original_training_set = CustomCIFAR100(root=cifar_100_root, transform=train_transform, train=True)
    whole_training_set = deepcopy(original_training_set)
    if args.mini:
        subsample_indices = subsample_instances(deepcopy(whole_training_set), prop_indices_to_subsample=0.2)
        whole_training_set = subsample_dataset(deepcopy(whole_training_set), subsample_indices)

    whole_idx = whole_training_set.uq_idxs

    class_idx_dic = {}
    for i in class_dic.keys():
        this_split = class_dic[i]
        this_idx = [x for x, t in enumerate(whole_training_set.targets) if t == i]
        shuffle(this_idx)
        class_idx_dic[i] = np.split(this_idx, [int(len(this_idx) * this_split[0]),
                                               int(len(this_idx) * (this_split[0] + this_split[1]))])

    # get overall indx list for each dataset
    step0_idx, step1_idx, step2_idx = [], [], []
    for i in class_idx_dic.keys():
        step0_idx.extend(class_idx_dic[i][0])
        step1_idx.extend(class_idx_dic[i][1])
        step2_idx.extend(class_idx_dic[i][2])

    shuffle(step0_idx)
    shuffle(step1_idx)
    shuffle(step2_idx)

    step0_dataset = subsample_dataset(deepcopy(whole_training_set), step0_idx)
    step1_dataset = subsample_dataset(deepcopy(whole_training_set), step1_idx)
    step2_dataset = subsample_dataset(deepcopy(whole_training_set), step2_idx)

    test_dataset = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)
    if args.mini:
        subsample_indices = subsample_instances(test_dataset, prop_indices_to_subsample=0.2)
        test_dataset = subsample_dataset(test_dataset, subsample_indices)

    step1_test_dataset = subsample_classes(deepcopy(test_dataset), include_classes=list(range(90)))

    all_datasets = {'step0_dataset': step0_dataset, 'step1_dataset': step1_dataset, 'step2_dataset': step2_dataset,
                    'test_dataset': {'step1_test_dataset': step1_test_dataset, 'step2_test_dataset': test_dataset}}

    return all_datasets

def incre_cifar100_exp2(train_transform=None, test_transform=None, args=None):
    class_dic = {}
    for i in range(100):
        if i < 20:
            class_dic[i] = [0.7, 0.3, 0]
        elif i < 40:
            class_dic[i] = [0, 0.7, 0.3]
        elif i < 60:
            class_dic[i] = [0, 0, 1]
        elif i < 100:
            class_dic[i] = [0, 0, 0]

    np.random.seed(0)

    # Get who data set
    original_training_set = CustomCIFAR100(root=cifar_100_root, transform=train_transform, train=True)
    whole_training_set = deepcopy(original_training_set)
    if args.mini:
        subsample_indices = subsample_instances(deepcopy(whole_training_set), prop_indices_to_subsample=0.4)
        whole_training_set = subsample_dataset(deepcopy(whole_training_set), subsample_indices)

    whole_idx = whole_training_set.uq_idxs

    class_idx_dic = {}
    for i in class_dic.keys():
        this_split = class_dic[i]
        this_idx = [x for x, t in enumerate(whole_training_set.targets) if t == i]
        shuffle(this_idx)
        class_idx_dic[i] = np.split(this_idx, [int(len(this_idx) * this_split[0]),
                                               int(len(this_idx) * (this_split[0] + this_split[1]))])

    # get overall indx list for each dataset
    step0_idx, step1_idx, step2_idx = [], [], []
    for i in class_idx_dic.keys():
        step0_idx.extend(class_idx_dic[i][0])
        step1_idx.extend(class_idx_dic[i][1])
        step2_idx.extend(class_idx_dic[i][2])

    shuffle(step0_idx)
    shuffle(step1_idx)
    shuffle(step2_idx)

    step0_dataset = subsample_dataset(deepcopy(whole_training_set), step0_idx)
    step1_dataset = subsample_dataset(deepcopy(whole_training_set), step1_idx)
    step2_dataset = subsample_dataset(deepcopy(whole_training_set), step2_idx)

    test_dataset = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)
    if args.mini:
        subsample_indices = subsample_instances(test_dataset, prop_indices_to_subsample=0.4)
        test_dataset = subsample_dataset(test_dataset, subsample_indices)

    step1_test_dataset = subsample_classes(deepcopy(test_dataset), include_classes=list(range(40)))
    step2_test_dataset = subsample_classes(deepcopy(test_dataset), include_classes=list(range(60)))

    all_datasets = {'step0_dataset': step0_dataset, 'step1_dataset': step1_dataset, 'step2_dataset': step2_dataset,
                    'test_dataset': {'step1_test_dataset': step1_test_dataset, 'step2_test_dataset': step2_test_dataset}}

    return all_datasets

if __name__ == '__main__':

    # x = get_cifar_100_datasets(None, None, split_train_val=False,
    #                      train_classes=range(80), prop_train_labels=0.5)

    x = incre_cifar10_exp1()

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing datasets overlap...')
    print(set.intersection(set(x['step0_dataset'].uq_idxs), set(x['step1_dataset'].uq_idxs)))
    print(set.intersection(set(x['step0_dataset'].uq_idxs), set(x['step2_dataset'].uq_idxs)))
    print(set.intersection(set(x['step1_dataset'].uq_idxs), set(x['step2_dataset'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['step0_dataset'].uq_idxs)) + len(set(x['step1_dataset'].uq_idxs)) + len(
        set(x['step2_dataset'].uq_idxs)))

    print(f'Num step0 Classes: {len(set(x["step0_dataset"].targets))}, {Counter(x["step0_dataset"].targets)}')
    print(f'Num step1 Classes: {len(set(x["step1_dataset"].targets))}, {Counter(x["step1_dataset"].targets)}')
    print(f'Num step2 Classes: {len(set(x["step2_dataset"].targets))}, {Counter(x["step2_dataset"].targets)}')
    print(f'Len step0 set: {len(x["step0_dataset"])}')
    print(f'Len step1 set: {len(x["step1_dataset"])}')
    print(f'Len step2 set: {len(x["step2_dataset"])}')

    # print('Printing labelled and unlabelled overlap...')
    # print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    # print('Printing total instances in train...')
    # print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    # print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    # print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    # print(f'Len labelled set: {len(x["train_labelled"])}')
    # print(f'Len unlabelled set: {len(x["train_unlabelled"])}')
