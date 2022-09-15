from torch.utils.data import Dataset
import numpy as np
import torch


def reshape_convert_2_torch(X, y):

    for i, item in enumerate(X):
        # reshape [X]
        if len(item.shape) < 4:
            X[i] = item.reshape((1,) + item.shape)

        # conver_2_torch of [X]
        X[i] = torch.from_numpy(X[i])

    print('X.shape: ', X.shape)
    print('y.shape: ', y.shape)

    # labels are ranged from 1 to 1251 thus we subtract 1
    y -= 1

    return X, torch.from_numpy(y).long()


class VoxCeleb1(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- *.npy
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, name, mode, transform=None, target_transform=None):
        super(VoxCeleb1, self).__init__()
        """
        VoxCeleb1 Class
        """

        data_ver = '2' if 'voxceleb2' in name else '1'
        path = '../data-link/audio/voxceleb' + data_ver + '/'

        if 'f64' in name:
            mfcc_path = 'f64_t96.npy'
            window_size = 96
        elif 'f32-t64' in name:
            mfcc_path = 'f32_t64.npy'
            window_size = 64
        elif 'f32-t32' in name:
            mfcc_path = 'f32_t32.npy'
            window_size = 32

        if mode == 'train':
            X = np.load(path + 'X_train_' + mfcc_path, allow_pickle=True)
            y = np.load(path + 'y_train_' + mfcc_path).astype(np.int16)
        elif mode == 'validation':
            X = np.load(path + 'X_val_' + mfcc_path, allow_pickle=True)
            y = np.load(path + 'y_val_' + mfcc_path).astype(np.int16)
        elif mode == 'test':
            X = np.load(path + 'X_test_' + mfcc_path, allow_pickle=True)
            y = np.load(path + 'y_test_' + mfcc_path).astype(np.int16)
        else:
            print("Unsupported Dataset")
            assert False

        self.X, self.y = reshape_convert_2_torch(X, y)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.window_size = window_size
        self.window_stride = window_size


    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X, y = self.X[index], self.y[index]
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)

        if self.mode == 'train':
            # NCHW = NC x # features x # sequence
            seq_len = X.size()[2]
            left_seq = np.random.randint(low=0, high=seq_len-self.window_size)
            right_seq = left_seq + self.window_size
            X = X[:,:,left_seq:right_seq]

        return X, y

    def get_data_dim(self):
        return list(self.X.size())

