from __future__ import print_function
import argparse, os, sys, json

current_path = os.path.abspath('.')
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import faiss

faiss.ProductQuantizer()

from src.data_loader import *
from src.utils import *

from src.models.simple_cnn import *
from src.models.resnet_models import *
from src.models.dscnn import *


def train_mtl(args, model, device, data, optimizer, lr_scheduler):
    # Joint MTL: Prepare Data
    for t, ncla in args.taskcla:
        print('*' * 100)
        print('Task {:2d} ({:s})'.format(t, data[t]['name']))
        print('*' * 100)

        if t == 0:
            xtrain = data[t]['train']['x']
            ytrain = data[t]['train']['y']
            task_t = t * torch.ones(xtrain.size(0)).int()
            task = task_t
        else:
            xtrain = torch.cat((xtrain, data[t]['train']['x']))
            ytrain = torch.cat((ytrain, data[t]['train']['y']))
            task_t = torch.cat((task_t, t * torch.ones(data[t]['train']['y'].size(0)).int()))
            task = task_t

    # Train w/ Test
    # args.best_acc = -0.1
    args.acc = np.zeros((len(args.taskcla), len(args.taskcla)), dtype=np.float32)
    args.acc_epoch = np.zeros((len(args.taskcla), len(args.taskcla)), dtype=np.int32)
    args.lss = np.zeros((len(args.taskcla), len(args.taskcla)), dtype=np.float32)
    args.lss_epoch = np.zeros((len(args.taskcla), len(args.taskcla)), dtype=np.int32)
    for epoch in range(args.num_epochs):
        print(f'Epoch: {epoch + 1:5d},    LR: {lr_scheduler.get_last_lr()}')
        args.epoch = epoch + 1
        train_epoch(args, model, device, task, xtrain, ytrain, optimizer, lr_scheduler)
        test(args, model, device, data)
        lr_scheduler.step()


def train_epoch(args, model, device, task, x, y, optimizer, lr_scheduler):
    model.train()

    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)

    num_batches = len(x) // args.batch_size
    j = 0
    # Loop batches
    for i in range(0, len(r), args.batch_size):

        if i + args.batch_size <= len(r):
            b = r[i:i + args.batch_size]
        else:
            b = r[i:]
        images, targets, t = x[b].to(device), y[b].to(device), task[b].to(device)

        # Forward
        outputs = model(images)[t]
        loss = regularized_nll_loss(args, model, outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        # Update parameters
        optimizer.step()


def test(args, model, device, data):

    for u in range(args.num_tasks):
        xtest = data[u]['test']['x'].to(device)
        ytest = data[u]['test']['y'].to(device)
        test_loss, test_acc = eval(args, model, device, u, xtest, ytest, debug=True)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.3f}% <<<'.format(u, data[u]['name'], test_loss,
                                                                                      100 * test_acc))
        if args.acc[args.num_tasks - 1, u] < test_acc:
            args.acc[args.num_tasks - 1, u] = test_acc
            args.acc_epoch[args.num_tasks - 1, u] = args.epoch
            # print(f'New Highest Accuracy: ({test_acc:.2f}%) at Epoch: ({args.epoch})\n')
            print('>>> New Highest Test Accuracy: Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.3f}%, epoch={:3d} <<<'.format(u, data[u]['name'], test_loss,
                                                                                          100 * test_acc, args.epoch))
        if args.lss[args.num_tasks - 1, u] > test_loss:
            args.lss[args.num_tasks - 1, u] = test_loss
            args.lss_epoch[args.num_tasks - 1, u] = args.epoch
            print('>>> New Lowest Test Loss: Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.3f}%, epoch={:3d} <<<'.format(u, data[u]['name'], test_loss,
                                                                                          100 * test_acc, args.epoch))

def eval(args, model, device, t, x, y, debug=False):
    model.eval()
    test_loss = 0
    correct = 0

    r = np.arange(x.size(0))
    r = torch.as_tensor(r, device=device, dtype=torch.int64)

    with torch.no_grad():
        num_batches = len(x) // args.test_batch_size
        # Loop batches
        for i in range(0, len(r), args.test_batch_size):
            if i + args.test_batch_size <= len(r):
                b = r[i:i + args.test_batch_size]
            else:
                b = r[i:]
            images, target = x[b].to(device), y[b].to(device)

            # Forward
            output = model(images)[t]
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= x.size(0)

    return test_loss / x.size(0), correct / x.size(0)


def prepare_model (args):
    if 'MTL' in args.model_arch:
        if 'LeNet' in args.model_arch:
            return LeNetMTL(args,
                            channels=[32, 32, 64, 64],
                            fcDims=[128,128],
                            pooling_type="max")

        elif 'DS-CNN' in args.model_arch or 'MicroNet' in args.model_arch:

            return None
        elif 'ResNet' in args.model_arch:
            return None

    if args.dataset == "mnist":
        return LeNet()
    elif args.dataset == "cifar10" or args.dataset == "svhn":
        if "LeNet" in args.model_arch:
            return CIFARLeNet()
        elif "ResNet18" in args.model_arch:
            return ResNet18(num_classes=args.config["classes"], pretrained=args.pretrained)
        elif "ResNet20" in args.model_arch:
            return PTCVResNet(name='resnet20_cifar10',pretrained=args.pretrained) if args.dataset == "cifar10" else PTCVResNet(name='resnet20_svhn',pretrained=args.pretrained)
        elif "MobileNetV2" in args.model_arch:
            pass
    elif args.dataset == "cifar100":
        if "LeNet" in args.model_arch:
            return CIFAR100LeNet()
        elif "ResNet18" in args.model_arch:
            return ResNet18(num_classes=args.config["classes"], pretrained=args.pretrained)
        elif "ResNet20" in args.model_arch:
            return PTCVResNet(name='resnet20_cifar100',pretrained=args.pretrained)
        elif "MobileNetV2" in args.model_arch:
            pass
    elif args.dataset == 'imagenet':
        if "ResNet18" in args.model_arch:
            return ResNet18(num_classes=args.config["classes"], pretrained=args.pretrained)
        elif "MobileNetV2" in args.model_arch:
            pass

    elif "emotion" in args.dataset:
        if "AudioCNN" in args.model_arch:
            return AudioCNN(channels=[32,32,64,64],
                         in_size=(args.config["features"],args.config["seq"] ),
                         num_classes=args.config["classes"],
                         pooling_type="avg")
        elif "DS-CNN" in args.model_arch or "MicroNet" in args.model_arch:
            return PTCVDSCNN(model_name=args.model_arch, in_channels=1,
                             num_classes=args.config["classes"],
                             init_block_kernel=(3,3),
                             pretrained=args.pretrained)
    elif 'urbansound' in args.dataset:
        if "AudioCNN" in args.model_arch:
            return AudioCNN(channels=[16, 16, 32, 32],
                        in_size=(args.config["features"], args.config["seq"]),
                        num_classes=args.config["classes"],
                        pooling_type="avg")
        elif "DS-CNN" in args.model_arch or "MicroNet" in args.model_arch:
            return PTCVDSCNN(model_name=args.model_arch, in_channels=1,
                             num_classes=args.config["classes"],
                             init_block_kernel=(3, 3),
                             pretrained=args.pretrained)
    elif "gsc" in args.dataset:
        if "AudioCNN" in args.model_arch:
            return AudioCNN(channels=[32, 32, 64, 64],
                        in_size=(args.config["features"], args.config["seq"]),
                        num_classes=args.config["classes"],
                        pooling_type="avg")
        elif "DS-CNN" in args.model_arch or "MicroNet" in args.model_arch:
            return PTCVDSCNN(model_name=args.model_arch, in_channels=1,
                             num_classes=args.config["classes"],
                             init_block_kernel=(10, 4),
                             pretrained=args.pretrained)

    elif "db6" in args.dataset:
        if "AudioCNN" in args.model_arch:
            return AudioCNN(channels=[32, 32, 64, 64],
                        in_size=(args.config["features"], args.config["seq"]),
                        num_classes=args.config["classes"],
                        pooling_type="avg")
        elif "DS-CNN" in args.model_arch or "MicroNet" in args.model_arch:
            return PTCVDSCNN(model_name=args.model_arch, in_channels=1,
                             num_classes=args.config["classes"],
                             init_block_kernel=(3, 3),
                             pretrained=args.pretrained)
    else:
        # for (1) HHAR, (2) PAMAP2, (3) Skoda, (4) DB2, (5) DB3 datasets, (6) VoxCeleb1, (7) GTSRB, GTSRB-VW,
        # (8) FashionMNIST, (9) STL10
        if "AudioCNN" in args.model_arch:
            return AudioCNN(channels=[16, 16, 32, 32],
                        in_size=(args.config["features"], args.config["seq"]),
                        num_classes=args.config["classes"],
                        pooling_type="avg")
        elif "DS-CNN" in args.model_arch or "MicroNet" in args.model_arch:
            return PTCVDSCNN(model_name=args.model_arch, in_channels=1,
                             num_classes=args.config["classes"],
                             init_block_kernel=(3, 3),
                             pretrained=args.pretrained)


def save_model (args, model):
    # torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
    #            f='./' + ckpt_path + '/{}'.format(args.ckpt_file))
    print(f'Best Accuracy: ({args.best_acc:.2f}%) at Epoch: ({args.best_acc_epoch})')
    torch.save(model.state_dict(), '../data/saved_model/' +
               args.model_arch + '_' +
               args.dataset + '_' +
               args.optimizer_name + '_' +
               args.lr_mode + '_' +
               use_mixup(args.mixup) +
               '.pt')
    # torch.save(model, f='../data/saved_model/' + args.model_arch + '_' + args.dataset + '.pt')


def save_model_mtl (args, model, data):
    for u in range(args.num_tasks):
        print(
            '>>> New Highest Test Accuracy: Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.3f}%, epoch={:3d} <<<'.format(
                u, data[u]['name'], args.lss[args.num_tasks - 1, u],
                100 * args.acc[args.num_tasks - 1, u], args.acc_epoch[args.num_tasks - 1, u]))

    torch.save(model.state_dict(), '../data/saved_model/mtl_' +
               args.model_arch + '_' +
               args.dataset + '_' +
               args.optimizer_name + '_' +
               args.lr_mode + '_' +
               use_mixup(args.mixup) +
               '.pt')


def prepare_trainer(model,
                    optimizer_name,
                    weight_decay,
                    momentum,
                    lr_mode,
                    lr,
                    lr_decay_period,
                    lr_decay_epoch,
                    lr_decay,
                    num_epochs,
                    warmup_epochs,
                    warmup_lr,
                    warmup_mode,
                    state_file_path):
    """
    Prepare trainer.

    Parameters:
    ----------
    net : Module
        Model.
    optimizer_name : str
        Name of optimizer.
    wd : float
        Weight decay rate.
    momentum : float
        Momentum value.
    lr_mode : str
        Learning rate scheduler mode.
    lr : float
        Learning rate.
    lr_decay_period : int
        Interval for periodic learning rate decays.
    lr_decay_epoch : str
        Epoches at which learning rate decays.
    lr_decay : float
        Decay rate of learning rate.
    num_epochs : int
        Number of training epochs.
    state_file_path : str
        Path for file with trainer state.

    Returns:
    -------
    Optimizer
        Optimizer.
    LRScheduler
        Learning rate scheduler.
    """

    optimizer_name = optimizer_name.lower()
    if (optimizer_name == "sgd") or (optimizer_name == "nag"):
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay,
                                    nesterov=(optimizer_name == "nag"))
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=lr,
                                     betas=(0.9, 0.999),
                                     weight_decay=weight_decay)
    else:
        raise ValueError("Usupported optimizer: {}".format(optimizer_name))

    lr_mode = lr_mode.lower()
    if lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in lr_decay_epoch.split(",")]
    if (lr_mode == "step") and (lr_decay_period != 0):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=lr_decay_period,
            gamma=lr_decay,
            last_epoch=-1)
    elif (lr_mode == "multistep") or ((lr_mode == "step") and (lr_decay_period == 0)):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=lr_decay_epoch,
            gamma=lr_decay,
            last_epoch=-1)
    elif lr_mode == "cosine" and warmup_epochs == 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=num_epochs)
    elif lr_mode == "cosine" and warmup_epochs != 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=warmup_epochs,
            T_mult=int(warmup_lr))
        # lr_scheduler_warmup = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer=optimizer,
        #     T_0=warmup_epochs,
        #     T_mult=warmup_lr)
    else:
        raise ValueError("Usupported lr_scheduler: {}".format(lr_mode))

    return optimizer, lr_scheduler

def init(parser):
    args = parser.parse_args()

    args.test_fold_l = json.loads(args.test_fold_l)
    args.channels = json.loads(args.channels)
    args.fcDims = json.loads(args.fcDims)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)

    return args, use_cuda, device, kwargs

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default="mnist",
                        metavar='D', help='training dataset (mnist or cifar10)')
    parser.add_argument('--model-arch', type=str, default="LeNet",
                        help='NN architecture (LeNet, AlexNet, ResNet, MobileNet)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--alpha', type=float, default=5e-4, metavar='L',
                        help='l2 norm weight (default: 5e-4)')
    parser.add_argument('--rho', type=float, default=1e-2, metavar='R',
                        help='cardinality weight (default: 1e-2)')
    parser.add_argument('--l1', default=False, action='store_true',
                        help='prune weights with l1 regularization instead of cardinality')
    parser.add_argument('--l2', default=False, action='store_true',
                        help='apply l2 regularization')
    parser.add_argument('--num_epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--num_re_epochs', type=int, default=3, metavar='R',
                        help='number of epochs to retrain (default: 3)')
    parser.add_argument('--shuffle', action="store_false", default=True)
    parser.add_argument('--channels', type=str, default='[32,32,64,64]')
    parser.add_argument('--fcDims', type=str, default='[128]')
    parser.add_argument('--kernels', type=str, default='[(3,3)]')

    parser.add_argument("--optimizer-name",type=str,default="nag",
                        help="optimizer name")
    parser.add_argument("--lr",type=float,default=0.1,
                        help="learning rate")
    parser.add_argument("--lr-mode",type=str,default="cosine",
                        help="learning rate scheduler mode. options are step, poly and cosine")
    parser.add_argument("--lr-decay",type=float,default=0.1,
                        help="decay rate of learning rate")
    parser.add_argument("--lr-decay-period",type=int,default=0,
                        help="interval for periodic learning rate decays. default is 0 to disable")
    parser.add_argument("--lr-decay-epoch",type=str,default="40,60",
                        help="epoches at which learning rate decays")
    parser.add_argument("--target-lr",type=float,default=1e-8,
                        help="ending learning rate")
    parser.add_argument("--poly-power",type=float,default=2,
                        help="power value for poly LR scheduler")
    parser.add_argument("--warmup-epochs",type=int,default=0,
                        help="number of warmup epochs")
    parser.add_argument("--warmup-lr",type=float,default=1e-8,
                        help="starting warmup learning rate")
    parser.add_argument("--warmup-mode",type=str,default="linear",
                        help="learning rate scheduler warmup mode. options are linear, poly and constant")
    parser.add_argument("--momentum",type=float,default=0.9,
                        help="momentum value for optimizer")
    parser.add_argument("--wd",type=float,default=0.0001,
                        help="weight decay rate")
    parser.add_argument("--gamma-wd-mult",type=float,default=1.0,
                        help="weight decay multiplier for batchnorm gamma")
    parser.add_argument("--beta-wd-mult",type=float,default=1.0,
                        help="weight decay multiplier for batchnorm beta")
    parser.add_argument("--bias-wd-mult",type=float,default=1.0,
                        help="weight decay multiplier for bias")
    parser.add_argument("--grad-clip",type=float,default=None,
                        help="max_norm for gradient clipping")
    parser.add_argument("--label-smoothing",action="store_true",
                        help="use label smoothing")

    parser.add_argument("--mixup",action="store_true",help="use mixup strategy")
    parser.add_argument("--mixup-alpha", type=float, default=1.0, help="mixup alpha value")
    parser.add_argument("--mixup-epoch-tail",type=int,default=10,
                        help="number of epochs without mixup at the end of training")

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--save-model-path', default="../data/saved_model/",
                        help='Path the trained model will be saved')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='For using pretrained CV models')

    parser.add_argument('--test_fold_l', type=str, default='[10]')
    parser.add_argument('--use_one_task', type=str, default='false')
    parser.add_argument('--exp_setup', type=str, default='')
    parser.add_argument('--subject_idx', type=int, default=None)
    parser.add_argument('--session', type=int, default=1)
    parser.add_argument('--test_vote', type=str, default=None)

    args, use_cuda, device, kwargs = init(parser)

    data,taskcla,inputsize = get_data_mtl(args=args, kwargs=kwargs)
    print('Input size =', inputsize, '\nTask info =', taskcla)
    args.num_tasks = len(taskcla)
    args.inputsize, args.taskcla = inputsize, taskcla

    model = prepare_model(args).to(device)

    optimizer, lr_scheduler = prepare_trainer(model=model,
                                            optimizer_name=args.optimizer_name,
                                            weight_decay=args.wd,
                                            momentum=args.momentum,
                                            lr_mode=args.lr_mode,
                                            lr=args.lr,
                                            lr_decay_period=args.lr_decay_period,
                                            lr_decay_epoch=args.lr_decay_epoch,
                                            lr_decay=args.lr_decay,
                                            num_epochs=args.num_epochs,
                                            warmup_epochs=args.warmup_epochs,
                                            warmup_lr=args.warmup_lr,
                                            warmup_mode=args.warmup_mode,
                                            state_file_path=None)

    train_mtl(args, model, device, data, optimizer, lr_scheduler)
    test(args, model, device, data)

    save_model_mtl(args, model, data)

if __name__ == "__main__":
    main()
