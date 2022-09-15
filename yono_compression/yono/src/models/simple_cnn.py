import math

import torch.nn as nn
import torch.nn.functional as F

# 3. make adjustable ResNet (resnet 20 50) for fine-tuning for each data (see difference between imagenetResnet and CIFARresnet)
# 4. found a net structure for each dataset
# 5. repeat 3.4. for mobileNetV1 mobileNetV2

def get_feature_seq_dim_nopadding(n_conv_layers, in_size, kernels=[(3,3)]):
    features_dim = in_size[0]
    seq_dim = in_size[1]

    for i in range(n_conv_layers):
        features_dim -= kernels[i][0]-1 if len(kernels) > 1 else kernels[0][0]-1
        seq_dim -= kernels[i][1]-1 if len(kernels) > 1 else kernels[0][1]-1
        features_dim = math.floor(features_dim / 2)
        seq_dim = math.floor(seq_dim / 2)

    return (features_dim, seq_dim)


def get_feature_seq_dim(n_conv_layers, in_size):
    features_dim = in_size[0]
    seq_dim = in_size[1]

    for i in range(1, n_conv_layers, 2):
        features_dim = math.floor(features_dim / 2)
        # features_dim = math.ceil(features_dim / 2)
        seq_dim = math.floor(seq_dim / 2)
        # seq_dim = math.ceil(seq_dim / 2)
    return (features_dim, seq_dim)



class LeNetMTL(nn.Module):
    def __init__(self,
                 args,
                 channels=[32,32,64,64],
                 fcDims=[128,128],
                 pooling_type='avg'
                 ):
        super(LeNetMTL, self).__init__()
        self.inputsize = args.inputsize
        self.taskcla = args.taskcla

        self.features = nn.Sequential()
        self.features.add_module("init_block",
             nn.Sequential(
                 nn.Conv2d(self.inputsize[0], channels[0], kernel_size=(3, 3), stride=1,padding=0),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(2, stride=2) if pooling_type == 'max' else nn.AvgPool2d(2, stride=2))
         )

        for i, channels_per_stage in enumerate(channels):
            if i == 0:
                continue
            self.features.add_module("stage{}".format(i),
                 nn.Sequential(
                     nn.Conv2d(channels[i-1], channels[i], kernel_size=(3, 3), stride=1,padding=0),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(2, stride=2) if pooling_type == 'max' else nn.AvgPool2d(2, stride=2))
             )

        self.dropout = nn.Dropout(0.5)
        features_dim, seq_dim = get_feature_seq_dim_nopadding(len(channels), self.inputsize[1:])

        self.decoder = nn.Sequential()
        self.decoder.add_module("init_decoder",
            nn.Sequential(
                nn.Linear(channels[-1] * features_dim * seq_dim, fcDims[0]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5))
        )

        if fcDims > 1:
            for i, channels_per_stage in enumerate(fcDims):
                if i == 0:
                    continue
                self.decoder.add_module("decode_stage{}".format(i),
                    nn.Sequential(
                        nn.Linear(fcDims[i-1], fcDims[i]),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5))
                )

        self.output = nn.ModuleList()
        for t, n in self.taskcla:
            self.output.append(nn.Linear(fcDims[-1], n))


    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        y = []
        for t, i in self.taskcla:
            y.append(self.output[t](x))
        return [F.log_softmax(yy, dim=1) for yy in y]


class AudioCNN(nn.Module):
    def __init__(self,
                 channels=[32,32,64,64],
                 in_channels=1,
                 in_size=(32, 32),
                 num_classes=10,
                 pooling_type='avg'
                 ):
        super(AudioCNN, self).__init__()

        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels,channels[0],kernel_size=(3, 3),stride=1,padding=1),
                        nn.BatchNorm2d(channels[0]),
                        nn.ReLU(inplace=True),
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(channels[0], channels[1], kernel_size=(3, 3), stride=1, padding=1),
                        nn.BatchNorm2d(channels[1]),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, stride=2) if pooling_type == 'max' else nn.AvgPool2d(2, stride=2)
                    )
        self.conv3 = nn.Sequential(
                        nn.Conv2d(channels[1], channels[2], kernel_size=(3, 3), stride=1, padding=1),
                        nn.BatchNorm2d(channels[2]),
                        nn.ReLU(inplace=True),
                    )
        self.conv4 = nn.Sequential(
                        nn.Conv2d(channels[2], channels[3], kernel_size=(3, 3), stride=1, padding=1),
                        nn.BatchNorm2d(channels[3]),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, stride=2) if pooling_type == 'max' else nn.AvgPool2d(2, stride=2)
                    )

        self.dropout = nn.Dropout(0.5)
        features_dim, seq_dim = get_feature_seq_dim(4, in_size)
        self.fc = nn.Linear(channels[3]*features_dim*seq_dim, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class VWNet(nn.Module):
    def __init__(self,
                 args,
                 in_channels=1,
                 in_size=(32, 32),
                 channels=[32, 32, 64, 64],
                 fcDims=[128, 128],
                 kernels=[(3,3)],
                 num_classes=10,
                 pooling_type='avg'
                 ):
        super(VWNet, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module("init_block",
             nn.Sequential(
                 nn.Conv2d(in_channels, channels[0], kernel_size=kernels[0], stride=1,padding=0),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(2, stride=2) if pooling_type == 'max' else nn.AvgPool2d(2, stride=2))
         )

        for i, channels_per_stage in enumerate(channels):
            if i == 0:
                continue
            self.features.add_module("stage{}".format(i),
                 nn.Sequential(
                     nn.Conv2d(channels[i-1], channels[i], kernel_size=kernels[i] if len(kernels)>1 else kernels[0], stride=1,padding=0),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(2, stride=2) if pooling_type == 'max' else nn.AvgPool2d(2, stride=2))
             )

        self.dropout = nn.Dropout(0.5)
        features_dim, seq_dim = get_feature_seq_dim_nopadding(len(channels), in_size, kernels)

        self.decoder = nn.Sequential()
        self.decoder.add_module("init_decoder",
            nn.Sequential(
                nn.Linear(channels[-1] * features_dim * seq_dim, fcDims[0]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5))
        )

        if len(fcDims) > 1:
            for i, channels_per_stage in enumerate(fcDims):
                if i == 0:
                    continue
                self.decoder.add_module("decode_stage{}".format(i),
                    nn.Sequential(
                        nn.Linear(fcDims[i-1], fcDims[i]),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5))
                )

        self.output = nn.Linear(fcDims[-1], num_classes)


    def forward(self, x):
        x = self.features(x)
        # print(x.size())
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CIFARLeNet(nn.Module):
    def __init__(self):
        super(CIFARLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CIFAR100LeNet(nn.Module):
    def __init__(self):
        super(CIFAR100LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500, 100)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
