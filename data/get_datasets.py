from data.data_utils import MergedDataset

from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets, incre_cifar10_exp1, incre_cifar100_exp1, incre_cifar100_exp2
from data.herbarium_19 import get_herbarium_datasets
from data.stanford_cars import get_scars_datasets
from data.imagenet import get_imagenet_100_datasets
from data.cub import get_cub_datasets
from data.fgvc_aircraft import get_aircraft_datasets

from data.cifar import subsample_classes as subsample_dataset_cifar
from data.herbarium_19 import subsample_classes as subsample_dataset_herb
from data.stanford_cars import subsample_classes as subsample_dataset_scars
from data.imagenet import subsample_classes as subsample_dataset_imagenet
from data.cub import subsample_classes as subsample_dataset_cub
from data.fgvc_aircraft import subsample_classes as subsample_dataset_air

from copy import deepcopy
import pickle
import os

from config import osr_split_dir

sub_sample_class_funcs = {
    'cifar10': subsample_dataset_cifar,
    'cifar100': subsample_dataset_cifar,
    'imagenet_100': subsample_dataset_imagenet,
    'herbarium_19': subsample_dataset_herb,
    'cub': subsample_dataset_cub,
    'aircraft': subsample_dataset_air,
    'scars': subsample_dataset_scars
}

get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'imagenet_100': get_imagenet_100_datasets,
    'herbarium_19': get_herbarium_datasets,
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets
}


def get_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                            train_classes=args.train_classes,
                            prop_train_labels=args.prop_train_labels,
                            split_train_val=False, args=args)

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform

    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets


def get_incre_datasets(dataset_name, train_transform, test_transform, args):
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    if dataset_name == 'cifar10':
        datasets = incre_cifar10_exp1(train_transform=train_transform, test_transform=test_transform, args=args)
    elif dataset_name == 'cifar100':
        if args.incre_exp == 1:
            datasets = incre_cifar100_exp1(train_transform=train_transform, test_transform=test_transform, args=args)
        elif args.incre_exp == 2:
            datasets = incre_cifar100_exp2(train_transform=train_transform, test_transform=test_transform, args=args)
    test_dataset = datasets['test_dataset']
    if args.step == 1:
        train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['step0_dataset']),
                                      unlabelled_dataset=deepcopy(datasets['step1_dataset']))
        unlabelled_train_examples_test = deepcopy(datasets['step1_dataset'])
        unlabelled_train_examples_test.transform = test_transform
        return train_dataset, test_dataset, unlabelled_train_examples_test, datasets
    elif args.step > 1:
        train_dataset = deepcopy(datasets[f'step{args.step}_dataset'])
        unlabelled_train_examples_test = deepcopy(datasets[f'step{args.step}_dataset'])
        unlabelled_train_examples_test. transform = test_transform
        return train_dataset, test_dataset, unlabelled_train_examples_test, datasets


def get_class_splits(args):

    # For FGVC datasets, optionally return bespoke splits
    if args.dataset_name in ('scars', 'cub', 'aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10':

        if args.incre_exp == 1:
            args.C_step0 = [0, 1, 2, 3, 4]
            args.C_step1 = [0,1,2,3,5,6,7]
            args.C_step2 = [0,1,4,7,8,9]
        else:
            args.image_size = 32
            args.train_classes = range(5)
            args.unlabeled_classes = range(5, 10)

    elif args.dataset_name == 'cifar100':
        if args.incre_exp == 1:
            args.C_step0 = list(range(80))
            args.C_step1 = list(range(10,90))
            args.C_step2 = list(range(20,100))
        elif args.incre_exp == 2:
            args.C_step0 = list(range(20))
            args.C_step1 = list(range(0,40))
            args.C_step2 = list(range(20,60))
        else:
            args.image_size = 32
            args.train_classes = range(80)
            args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'tinyimagenet':

        args.image_size = 64
        args.train_classes = range(100)
        args.unlabeled_classes = range(100, 200)

    elif args.dataset_name == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']

    elif args.dataset_name == 'imagenet_100':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'scars':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

    elif args.dataset_name == 'aircraft':

        args.image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cub':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(101)
            args.unlabeled_classes = range(101, 201)

    elif args.dataset_name == 'chinese_traffic_signs':

        args.image_size = 224
        args.train_classes = range(28)
        args.unlabeled_classes = range(28, 56)

    else:

        raise NotImplementedError

    return args