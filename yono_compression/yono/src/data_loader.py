import os,sys
import copy

import numpy as np

import torch
from torchvision import datasets, transforms
from src.datasets.singletask_data import get_singletask_data
from src.datasets.constants import *
from src.datasets.datasets_image import *
import src.datasets.voxceleb as vc
from sklearn.utils import shuffle
from src.datasets.utils import get_mean_std


# dataloader for image, sound, har, sEMG, GAIT
# 5 image datasets:
### MNIST, CIFAR10, CIFAR100, SVHN, IMAGENET

def get_data_loaders(args, kwargs=None):
    args.config = DATASET_CONFIGS[args.dataset]
    # dataloader for images
    # get_mean_std(datasets.STL10('../data-link/image', split="train", download=True, transform=transforms.Compose([
    #     transforms.ToTensor(), ])))

    if 'vw' in args.dataset:
        train_dataset, test_dataset = get_singletask_data(args, transform=None, target_transform=None, subject_idx=None,exp_setup=None, test_fold_l=None)
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=args.shuffle, batch_size=args.batch_size,**kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.test_batch_size,**kwargs)
    elif args.dataset == "mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data-link/image', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data-link/image', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'fashion-mnist':
        # mean_rgb, std_rgb = (0.2860,), (0.3205,)
        mean_rgb, std_rgb = (0.1307,), (0.3081,)
        transform_train = transforms.Compose([
            transforms.RandomCrop(size=args.config['features'], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_rgb, std=std_rgb)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_rgb, std=std_rgb)
        ])

        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data-link/image', train=True, download=True,
                             transform=transform_train), shuffle=args.shuffle, batch_size=args.batch_size, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data-link/image', train=False, download=True,
                             transform=transform_test), shuffle=False, batch_size=args.test_batch_size, **kwargs)
        # get_mean_std(datasets.FashionMNIST('../data-link/image', train=True, download=True,transform=transforms.Compose([
        #     transforms.ToTensor(),])))
        # pass

    elif args.dataset == "imagenet":
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageNet('../data-link/image', split='train', download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406),
                                                      (0.229, 0.224, 0.225))
                             ])), shuffle=args.shuffle, batch_size=args.batch_size, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageNet('../data-link/image', split='val', download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406),
                                                      (0.229, 0.224, 0.225))
                             ])), shuffle=False, batch_size=args.test_batch_size, **kwargs)

    elif "cifar" in args.dataset or "svhn" in args.dataset or "gtsrb" in args.dataset or "stl10" in args.dataset:
        # mean_rgb_cifar10 = (0.49139968, 0.48215827, 0.44653124)
        # std_rgb_cifar10 = (0.24703233, 0.24348505, 0.26158768)
        # mean_rgb_cifar100 = (0.5070746, 0.48654896, 0.44091788)
        # std_rgb_cifar100 = (0.26733422, 0.25643846, 0.27615058)
        # mean_rgb_svhn = (0.43768218, 0.44376934, 0.47280428)
        # std_rgb_svhn = (0.1980301, 0.2010157, 0.19703591)
        mean_rgb = (0.4914, 0.4822, 0.4465)
        std_rgb = (0.2023, 0.1994, 0.2010)
        if "gtsrb" in args.dataset:
            mean_rgb = [0.3398, 0.3117, 0.3210]
            std_rgb = [0.2755, 0.2647, 0.2712]
        elif "stl10" in args.dataset:
            mean_rgb = [0.4467, 0.4398, 0.4066]
            std_rgb = [0.2242, 0.2215, 0.2239]

        transform_train = transforms.Compose([
                                        transforms.RandomCrop(size=args.config['features'],padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean_rgb,std=std_rgb)
                                    ])
        transform_test = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=mean_rgb, std=std_rgb)
                             ])

        if args.dataset == "cifar10":
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data-link/image', train=True, download=True,
                                 transform=transform_train), shuffle=args.shuffle, batch_size=args.batch_size, **kwargs)

            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data-link/image', train=False, download=True,
                                 transform=transform_test), shuffle=False, batch_size=args.test_batch_size, **kwargs)

        elif args.dataset == "cifar100":
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data-link/image', train=True, download=True,
                                 transform=transform_train), shuffle=args.shuffle, batch_size=args.batch_size, **kwargs)

            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data-link/image', train=False, download=True,
                                 transform=transform_test), shuffle=False, batch_size=args.test_batch_size, **kwargs)

        elif args.dataset == "svhn":
            train_loader = torch.utils.data.DataLoader(
                datasets.SVHN('../data-link/image/svhn', split='train', download=True,
                                 transform=transform_train), shuffle=args.shuffle, batch_size=args.batch_size, **kwargs)

            test_loader = torch.utils.data.DataLoader(
                datasets.SVHN('../data-link/image/svhn', split='test', download=True,
                                 transform=transform_test), shuffle=False, batch_size=args.test_batch_size, **kwargs)
        elif args.dataset == "gtsrb":
            train_loader = torch.utils.data.DataLoader(
                TrafficSigns('../data-link/image/gtsrb', train=True, download=True, transform=transform_train),
                shuffle=args.shuffle, batch_size=args.batch_size, **kwargs)

            test_loader = torch.utils.data.DataLoader(
                TrafficSigns('../data-link/image/gtsrb', train=False, download=True, transform=transform_test),
                shuffle=False, batch_size=args.test_batch_size, **kwargs)
        elif args.dataset == "stl10":
            train_loader = torch.utils.data.DataLoader(
                datasets.STL10('../data-link/image/', split='train', download=True,
                              transform=transform_train), shuffle=args.shuffle, batch_size=args.batch_size, **kwargs)

            test_loader = torch.utils.data.DataLoader(
                datasets.STL10('../data-link/image/', split='test', download=True,
                              transform=transform_test), shuffle=False, batch_size=args.test_batch_size, **kwargs)

    #####
    ##### dataloader for sound #####
    elif "emotion" in args.dataset or "urbansound8k" in args.dataset or "gsc" in args.dataset:
        train_dataset, test_dataset = get_singletask_data(args,transform=None,target_transform=None,subject_idx=None,exp_setup=None,test_fold_l=None)
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=args.shuffle, batch_size=args.batch_size, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.test_batch_size, **kwargs)

    elif "voxceleb1" in args.dataset:
        train_dataset, test_dataset = vc.VoxCeleb1(name=args.dataset, mode='train'), vc.VoxCeleb1(name=args.dataset, mode='test')
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=args.shuffle, batch_size=args.batch_size,**kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.test_batch_size,**kwargs)

    #####
    ##### dataloader for HAR ####
    elif "hhar-noaug" in args.dataset or "pamap2" in args.dataset or "skoda" in args.dataset:
        train_dataset, test_dataset = get_singletask_data(args,transform=None,target_transform=None,subject_idx=None,exp_setup=None,test_fold_l=None)
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=args.shuffle, batch_size=args.batch_size, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.test_batch_size, **kwargs)

    #####
    ##### dataloader for sEMG ####
    elif "ninapro-db" in args.dataset or "emgshatilov" in args.dataset:
        train_dataset, test_dataset = get_singletask_data(args, transform=None, target_transform=None,
                                                          subject_idx=args.subject_idx,
                                                          exp_setup=args.exp_setup, test_fold_l=None)
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=args.shuffle, batch_size=args.batch_size, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.test_batch_size, **kwargs)

    #####
    ##### dataloader for GAIT ####
    elif "whugait" in args.dataset:
        pass
    elif "idnet" in args.dataset:
        pass

    return train_loader, test_loader


# Task Order
# 1: MNIST
# 2: Speech Commands
# 3: HHAR
# 4: GTSRB

def get_data_mtl(args, pc_valid=0.15, fixed_order=True, selected_tasks=['mnist', 'gscv2-vw-c35', 'hhar-noaug'], n_in_channels=1, kwargs=None):
    data = {}
    taskcla = []

    # selected_tasks = np.arange(3)
    if not fixed_order:
        selected_tasks = list(shuffle(selected_tasks, random_state=args.seed))
    print('Task order =', selected_tasks)

    # decide expanded input dim to match the input dim of all datasets
    size = [n_in_channels,1,1]
    for n, dname in enumerate(selected_tasks):
        if DATASET_CONFIGS[dname]['features'] > size[1]:
            size[1] = DATASET_CONFIGS[dname]['features']
        if DATASET_CONFIGS[dname]['seq'] > size[2]:
            size[2] = DATASET_CONFIGS[dname]['seq']

    path = os.path.join('../data-link', 'mtl')
    if not os.path.isdir(path):
        # os.makedirs(path)
        # Pre-load
        for n, dname in enumerate(selected_tasks):
            arg = copy.deepcopy(args)
            arg.dataset = dname
            arg.batch_size = 1
            arg.test_barch_size = 1
            arg.shuffle = False
            train_loader, test_loader = get_data_loaders(arg)
            data[n] = {}
            data[n]['name'] = dname
            data[n]['ncla'] = DATASET_CONFIGS[dname]['classes']
            data[n]['train'] = {'x': [], 'y': []}
            for image, target in train_loader:
                image = image.expand(1, size[0], size[1], size[2])  # Create 3 equal channels
                data[n]['train']['x'].append(image)
                data[n]['train']['y'].append(target.numpy()[0])
            data[n]['test'] = {'x': [], 'y': []}
            for image, target in test_loader:
                image = image.expand(1, size[0], size[1], size[2])  # Create 3 equal channels
                data[n]['test']['x'].append(image)
                data[n]['test']['y'].append(target.numpy()[0])

            # "Unify" and save
            for s in ['train', 'test']:
                data[n][s]['x'] = torch.stack(data[n][s]['x']).view(-1, size[0], size[1], size[2])
                data[n][s]['y'] = torch.LongTensor(np.array(data[n][s]['y'], dtype=int)).view(-1)

    # Validation
    # for t in data.keys():
    #     r = np.arange(data[t]['train']['x'].size(0))
    #     r = np.array(shuffle(r, random_state=args.seed), dtype=int)
    #     nvalid = int(pc_valid * len(r))
    #     ivalid = torch.LongTensor(r[:nvalid])
    #     itrain = torch.LongTensor(r[nvalid:])
    #     data[t]['valid'] = {}
    #     data[t]['valid']['x'] = data[t]['train']['x'][ivalid].clone()
    #     data[t]['valid']['y'] = data[t]['train']['y'][ivalid].clone()
    #     data[t]['train']['x'] = data[t]['train']['x'][itrain].clone()
    #     data[t]['train']['y'] = data[t]['train']['y'][itrain].clone()

    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data,taskcla,size

