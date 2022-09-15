import torch
import numpy as np
from scipy.io import loadmat
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from collections import Counter
from src.datasets.sliding_window import sliding_window

def get_mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset,
                             batch_size=10,
                             num_workers=0,
                             shuffle=False)

    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std

def reshape(x):
    if len(x.shape) >= 4:
        return x
    else:
        return x.reshape((x.shape[0],) + (1,) + x.shape[1:])

def convert_2_torch(X, y):
    print('X.shape: ', X.shape)
    print('y.shape: ', y.shape)

    return torch.from_numpy(X), torch.from_numpy(y).long()

def ensure_label_range(y):
    y2label = {
        1:0,
        3:1,
        4:2,
        6:3,
        9:4,
        10:5,
        11:6
               }
    for i, v in enumerate(y):
        y[i] = y2label[v]

def custom_normalize(X):
    X_shape = X.shape
    X_new = np.reshape(X, (X_shape[0], -1))
    X_new = StandardScaler().fit_transform(X_new)
    X_new = np.reshape(X_new, (X_shape[0], X_shape[1], X_shape[2]))
    return X_new

def opp_sliding_window(data_x, data_y, ws, ss):
    """
    Obtaining the windowed data from the HAR data
    :param data_x: sensory data
    :param data_y: labels
    :param ws: window size
    :param ss: stride
    :return: windows from the sensory data (based on window size and stride)
    """
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.reshape(data_y, (len(data_y), ))  # Just making it a vector if it was a 2D matrix
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


def load_and_segment_data(name, config=None, subject_idx=None, exp_setup=None, test_fold_l=[10], args=None):
    # default window size would be 1s, and stride size is 50%.
    if config is not None:
        window_size = config['seq']
        if args is not None:
            if 'cnn' in args.cls_type and 'ninapro-db2-c10' in name:
                stride_size = 2
            elif 'db6' in name:
                stride_size = int(window_size / 4)
            else:
                stride_size = int(window_size / 2)
        else:
            stride_size = int(window_size / 2)

    if name == 'opportunity':
        path = '../DeepConvLSTM-master/'
        X_train = np.load(path + "data/opp_X_train.npy")
        X_test = np.load(path + "data/opp_X_test.npy")
        y_train = np.load(path + "data/opp_y_train.npy")
        y_test = np.load(path + "data/opp_y_test.npy")
        # exclude null class
        X_train = X_train[y_train[:] > 0]
        y_train = y_train[y_train[:] > 0]
        y_train = y_train[:] - 1
        X_test = X_test[y_test[:] > 0]
        y_test = y_test[y_test[:] > 0]
        y_test = y_test[:] - 1

    elif name[:4] == 'nina':
        path = '../data-link/emg/ninapro/'

        if 'ninapro-db2-c50' in name:
            full_path = path + 'db2/processed_c50_2000Hz/'
            if 'auth' not in name:  # normal GR task
                data_raw = load_dataset_ninapro(full_path, name, subject_idx, exp_setup)
            else:  # User Identification or User Authentification task
                data_raw = load_dataset_ninapro_auth(full_path, name)
            X_train, y_train = data_raw['train_data'], data_raw['train_labels']
            X_eval, y_eval = data_raw['val_data'], data_raw['val_labels']
            X_test, y_test = data_raw['test_data'], data_raw['test_labels']
            X_train = np.concatenate((X_train, X_eval), axis=0)
            y_train = np.concatenate((y_train, y_eval), axis=0)
            return X_train, y_train, X_test, y_test
        elif name[:11] == 'ninapro-db2':
            full_path = path + 'db2/processed_2/'
        elif name[:11] == 'ninapro-db3':
            full_path = path + 'db3/processed_2/'
        elif name[:11] == 'ninapro-db6':
            full_path = path + 'db6/processed_c7_f8_200Hz/'
        else:
            full_path = ''

        if 'db6' in name:
            if 'auth' not in name:  # normal GR task
                X, y = load_dataset_ninapro_db6(full_path, name, subject_idx, exp_setup)
            else:  # User Identification or User Authentification task
                X, y = load_dataset_ninapro_db6_auth(full_path, name)
            # Obtaining the segmented data
            X, y = opp_sliding_window(X, y, window_size, stride_size)
            # shuffle since all labels are in increasing order
            X, y = shuffle(X, y, random_state=0)
            # ensure all labels in range [0,num_classes - 1]
            ensure_label_range(y)
            # split train, val, test set
            cut_train_th, cut_eval_th = 8, 9
            cut_train = int(len(X) / 10) * cut_train_th
            cut_eval = int(len(X) / 10) * cut_eval_th

            X_train = X[:cut_train]
            y_train = y[:cut_train]
            X_eval = X[cut_train:cut_eval]
            y_eval = y[cut_train:cut_eval]
            X_test = X[cut_eval:]
            y_test = y[cut_eval:]

            # Stacking train and eval sets
            X_train = np.concatenate((X_train, X_eval), axis=0)
            y_train = np.concatenate((y_train, y_eval), axis=0)

            # normalize
            #X_train = custom_normalize(X_train)
            #X_test = custom_normalize(X_test)

            return X_train.astype(np.float32), y_train.astype(np.uint8), X_test.astype(np.float32), y_test.astype(
                np.uint8)
        else:
            if 'auth' not in name:  # normal GR task
                data_raw = load_dataset_ninapro(full_path, name, subject_idx, exp_setup)
            else:  # User Identification or User Authentification task
                data_raw = load_dataset_ninapro_auth(full_path, name)
        # Obtaining the segmented data
        X_train, y_train = opp_sliding_window(data_raw['train' + '_data'], data_raw['train' + '_labels'],
                                              window_size, stride_size)
        X_eval, y_eval = opp_sliding_window(data_raw['val' + '_data'], data_raw['val' + '_labels'],
                                            window_size, stride_size)
        X_test, y_test = opp_sliding_window(data_raw['test' + '_data'], data_raw['test' + '_labels'],
                                            window_size, stride_size)
        # Stacking train and eval sets
        X_train = np.concatenate((X_train, X_eval), axis=0)
        y_train = np.concatenate((y_train, y_eval), axis=0)

    elif name[:4] == 'emot':
        path = '../data-link/audio/emotion/Splits/Emotions-all/'
        full_path = path + name + '.mat'

        # in audio dataset, we don't do overlapping so we can save more steps in LSTMs.
        stride_size = window_size

        data_raw = load_dataset(full_path)
        # Obtaining the segmented data
        X_train, y_train = opp_sliding_window(data_raw['train' + '_data'], data_raw['train' + '_labels'],
                                              window_size, stride_size)
        X_eval, y_eval = opp_sliding_window(data_raw['val' + '_data'], data_raw['val' + '_labels'],
                                            window_size, stride_size)
        X_test, y_test = opp_sliding_window(data_raw['test' + '_data'], data_raw['test' + '_labels'],
                                            window_size, stride_size)
        # Stacking train and eval sets
        X_train = np.concatenate((X_train, X_eval), axis=0)
        y_train = np.concatenate((y_train, y_eval), axis=0)

        # np.save(path+name+'_X_train', X_train)
        # np.save(path+name+'_y_train', y_train.astype(np.uint8))
        # np.save(path+name+'_X_test', X_test)
        # np.save(path+name+'_y_test', y_test.astype(np.uint8))
        # exit(0)

    elif 'urbansound8k' in name:
        path = '../data-link/audio/urbansound/UrbanSound8K/'
        if 'LMCST-1s-fullhop-45f' in name:
            path += 'audio-LMCST-1s-fullhop-45f/'
        elif 'LMCST-1s-45f' in name:
            path += 'audio-LMCST-1s-45f/'
        elif 'LMCST-4s' in name:
            path += 'audio-LMCST-4s/'
        elif 'LM' in name:
            path += 'audio-LMCST/'
        else:
            path += 'audio/'

        X_train = np.load(path + "urbansound8k-LMCST-long-randfold_X_train.npy")
        y_train = np.load(path + "urbansound8k-LMCST-long-randfold_y_train.npy")
        X_test = np.load(path + "urbansound8k-LMCST-long-randfold_X_test.npy")
        y_test = np.load(path + "urbansound8k-LMCST-long-randfold_y_test.npy")
        return X_train.astype(np.float32), y_train.astype(np.uint8), X_test.astype(np.float32), y_test.astype(np.uint8)
        X_test = []
        y_test = []
        X_train = []
        y_train = []

        if 'randfold' in name:
            X = np.array([])
            y = np.array([])
            for i in range(1, 11):
                if len(X) == 0:
                    X = np.load(path + "fold" + str(i) + "-X-mfcc.npy")
                    y = np.load(path + "fold" + str(i) + "-y-mfcc.npy")
                else:
                    X_temp = np.load(path + "fold" + str(i) + "-X-mfcc.npy")
                    y_temp = np.load(path + "fold" + str(i) + "-y-mfcc.npy")
                    X = np.concatenate((X, X_temp), axis=0)
                    y = np.concatenate((y, y_temp), axis=0)
            if len(test_fold_l) == 0:
                cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
                for train_index, test_index in cv.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    break
            else:
                test_fold = test_fold_l[0]
                X, y = shuffle(X, y, random_state=0)
                cv = KFold(n_splits=10, random_state=0)
                fold = 0
                for train_index, test_index in cv.split(X, y):
                    fold += 1
                    if fold == test_fold:
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]

        else:
            for i in range(1, 11):
                if i in test_fold_l:
                    if len(X_test) == 0:  # if X_test is empty
                        X_test = np.load(path + "fold" + str(i) + "-X-mfcc.npy")
                        y_test = np.load(path + "fold" + str(i) + "-y-mfcc.npy")
                    else:  # if X_test is not empty
                        X_temp = np.load(path + "fold" + str(i) + "-X-mfcc.npy")
                        y_temp = np.load(path + "fold" + str(i) + "-y-mfcc.npy")
                        X_test = np.concatenate((X_test, X_temp), axis=0)
                        y_test = np.concatenate((y_test, y_temp), axis=0)
                else:
                    if len(X_train) == 0:  # if X_train is empty
                        X_train = np.load(path + "fold" + str(i) + "-X-mfcc.npy")
                        y_train = np.load(path + "fold" + str(i) + "-y-mfcc.npy")
                    else:  # if X_train is not empty
                        X_temp = np.load(path + "fold" + str(i) + "-X-mfcc.npy")
                        y_temp = np.load(path + "fold" + str(i) + "-y-mfcc.npy")
                        X_train = np.concatenate((X_train, X_temp), axis=0)
                        y_train = np.concatenate((y_train, y_temp), axis=0)

        if 'long' in name or '1s' in name:
            print(config)
            X_train = X_train[:, :config['features'], :config['seq']]
            X_test = X_test[:, :config['features'], :config['seq']]
            X_train = np.transpose(X_train, (0, 2, 1))
            X_test = np.transpose(X_test, (0, 2, 1))
        elif 'augment' in name or 'LMCST' in name:
            X_train, y_train = augment_urbansound(X_train, y_train, config['seq'])
            if 'vote' not in name:
                X_test, y_test = augment_urbansound(X_test, y_test, config['seq'])
            else:  # if we do voting scheme in test
                X_test = np.transpose(X_test, (0, 2, 1)).astype(np.float32)
                y_test = y_test.astype(np.uint8)
        elif 'LM' in name:
            X_train = X_train[:, :config['features'], :config['seq']]
            X_test = X_test[:, :config['features'], :config['seq']]
            X_train = np.transpose(X_train, (0, 2, 1))
            X_test = np.transpose(X_test, (0, 2, 1))

        if 'shuffle' in name:
            X_train, y_train = shuffle(X_train, y_train, random_state=0)

        # for i in range(config['classes']):
        #     X_temp = X_train[y_train == i]
        #     np.save(path+name+'_X_train_'+str(i), X_temp)
        # np.save(path+name+'_X_train', X_train)
        # np.save(path+name+'_y_train', y_train.astype(np.uint8))
        # np.save(path+name+'_X_test', X_test)
        # np.save(path+name+'_y_test', y_test.astype(np.uint8))
        # exit(0)

    elif 'vw' in name:
        path = '../data-link/image/vw/'
        if 'cifar10' in name:
            X_train = np.load(path + 'cifar10_train_data.npy')
            y_train = np.load(path + 'cifar10_train_label.npy')
            y_train = np.argmax(y_train, axis=1).astype(np.uint8)
            X_test = np.load(path + 'cifar10_test_data.npy')
            y_test = np.load(path + 'cifar10_test_label.npy')
            y_test = np.argmax(y_test, axis=1).astype(np.uint8)
            # transpose to make the input dim [NCHW]
            X_train = np.transpose(X_train, (0, 3, 1, 2))
            X_test = np.transpose(X_test, (0, 3, 1, 2))
            return X_train.astype(np.float32), y_train.astype(np.uint8), X_test.astype(np.float32), y_test.astype(
                np.uint8)
        elif 'svhn' in name:
            X_train = np.load(path + 'svhn_train_data.npy')
            y_train = np.load(path + 'svhn_train_label.npy')
            y_train = np.argmax(y_train, axis=1).astype(np.uint8)
            X_validation = np.load(path + 'svhn_validation_data.npy')
            y_validation = np.load(path + 'svhn_validation_label.npy')
            y_validation = np.argmax(y_validation, axis=1).astype(np.uint8)
            X_test = np.load(path + 'svhn_test_data.npy')
            y_test = np.load(path + 'svhn_test_label.npy')
            y_test = np.argmax(y_test, axis=1).astype(np.uint8)
            # Stacking train and eval sets
            X_train = np.concatenate((X_train, X_validation), axis=0)
            y_train = np.concatenate((y_train, y_validation), axis=0)
            # transpose to make the input dim [NCHW]
            X_train = np.transpose(X_train, (0, 3, 1, 2))
            X_test = np.transpose(X_test, (0, 3, 1, 2))
            return X_train.astype(np.float32), y_train.astype(np.uint8), X_test.astype(np.float32), y_test.astype(
                np.uint8)
        elif 'gtsrb' in name:
            X_train = np.load(path + 'GTSRB_train_data.npy')
            y_train = np.load(path + 'GTSRB_train_label.npy')
            y_train = np.argmax(y_train, axis=1).astype(np.uint8)
            X_test = np.load(path + 'GTSRB_test_data.npy')
            y_test = np.load(path + 'GTSRB_test_label.npy')
            y_test = np.argmax(y_test, axis=1).astype(np.uint8)
        elif 'gsc' in name:
            path = '../data-link/audio/google_speech/vw_gscv2/'
            X_train = np.load(path + 'GSC_v2_train_data.npy')
            y_train = np.load(path + 'GSC_v2_train_label.npy')
            y_train = np.argmax(y_train, axis=1).astype(np.uint8)
            X_eval = np.load(path + 'GSC_v2_validation_data.npy')
            y_eval = np.load(path + 'GSC_v2_validation_label.npy')
            y_eval = np.argmax(y_eval, axis=1).astype(np.uint8)
            X_test = np.load(path + 'GSC_v2_test_data.npy')
            y_test = np.load(path + 'GSC_v2_test_label.npy')
            y_test = np.argmax(y_test, axis=1).astype(np.uint8)
            # Stacking train and eval sets
            X_train = np.concatenate((X_train, X_eval), axis=0)
            y_train = np.concatenate((y_train, y_eval), axis=0)

        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)

        X_train = np.reshape(X_train, (X_train.shape[0], config['features'], config['seq']))
        X_test = np.reshape(X_test, (X_test.shape[0], config['features'], config['seq']))


    elif 'gsc' in name:
        data_ver = 'v2' if 'v2' in name else 'v1'
        path = '../data-link/audio/google_speech/npy_gsc' + data_ver + '/'
        mfcc_path = 'mfcc10_t49' if 'mfcc10' in name else 'mfcc40_t49'
        X_train = np.load(path+'X_train_'+mfcc_path+'.npy')
        y_train = np.load(path + 'y_train_' + mfcc_path + '.npy')
        X_eval = np.load(path + 'X_val_' + mfcc_path + '.npy')
        y_eval = np.load(path + 'y_val_' + mfcc_path + '.npy')
        X_test = np.load(path + 'X_test_' + mfcc_path + '.npy')
        y_test = np.load(path + 'y_test_' + mfcc_path + '.npy')

        # Stacking train and eval sets
        X_train = np.concatenate((X_train, X_eval), axis=0)
        y_train = np.concatenate((y_train, y_eval), axis=0)

    elif name[:4] == 'hhar':
        path = '../data-link/imu/hhar/'
        if name == 'hhar-noaug':
            full_path = path + 'noaug_leave_one_user_out/'
        elif name == 'hhar-aug':
            full_path = path + 'aug_leave_one_user_out/'
        elif name == 'hhar-raw':
            full_path = path + 'raw_leave_one_user_out/'
        else:
            full_path = ''
        X_train = np.load(full_path + "a_train_X.npy").astype(np.float32)
        X_test = np.load(full_path + "a_test_X.npy").astype(np.float32)
        y_train = np.load(full_path + "a_train_y.npy").astype(np.uint8)
        y_test = np.load(full_path + "a_test_y.npy").astype(np.uint8)
    else:
        path = '../data-link/imu/data_harish/'
        if name == 'pamap2':
            full_path = path + 'pamap2/PAMAP2.mat'
        elif name == 'skoda':
            full_path = path + 'skoda/Skoda_nonull.mat'
        elif name == 'usc-had':
            full_path = path + 'usc-had/usc-had.mat'
        elif name == 'opp_thomas':
            full_path = path + 'opportunity/opportunity_hammerla.mat'
        else:
            full_path = ''
        data_raw = load_dataset(full_path)
        # Obtaining the segmented data
        X_train, y_train = opp_sliding_window(data_raw['train' + '_data'], data_raw['train' + '_labels'],
                                              window_size, stride_size)
        X_eval, y_eval = opp_sliding_window(data_raw['val' + '_data'], data_raw['val' + '_labels'],
                                            window_size, stride_size)
        X_test, y_test = opp_sliding_window(data_raw['test' + '_data'], data_raw['test' + '_labels'],
                                            window_size, stride_size)
        # Stacking train and eval sets
        X_train = np.concatenate((X_train, X_eval), axis=0)
        y_train = np.concatenate((y_train, y_eval), axis=0)

    return X_train.astype(np.float32), y_train.astype(np.uint8), X_test.astype(np.float32), y_test.astype(np.uint8)

def get_db6_DxTx(session=1):
    D, T = 1, 1
    if session % 2 == 1:
        T = 1
    else:
        T = 2
    if session <= 2:
        D = 1
    elif session <= 4:
        D = 2
    elif session <= 6:
        D = 3
    elif session <= 8:
        D = 4
    elif session <= 10:
        D = 5
    else:
        assert("wrong session info: session needs to be from 1 to 10, both inclusive")
    return '_D'+str(D)+'_T'+str(T)+'_'

def load_dataset_ninapro_db6(full_path, filename, subject_idx=None, exp_setup=None, session=1):
    X, y = [], []
    if exp_setup == 'per-subject':
        for i in range(2):
            session += i
            data = loadmat(full_path + 'S' + str(subject_idx) + get_db6_DxTx(session) + '200Hz.mat')
            X.extend(data['emg'])
            y.extend(data['restimulus'][0])

    elif exp_setup == 'leave-one-user-out':
        pass

    return np.array(X).astype(np.float32), np.array(y).astype(np.uint8)

def load_dataset_ninapro(full_path, filename, subject_idx=None, exp_setup=None):
    # target_gest_dict = {
    # 13: 0, 14:1, 10:2, 9:3, 5:4, 30:5, 34:6, 37:7, 32:8, 21:9
    # }
    target_gest_dict = {
        13: 0, 14: 1, 12: 2, 11: 3, 5: 4, 33: 5, 34: 6, 19: 7, 32: 8, 21: 9
    }
    target_user_set_db2 = {11, 31, 17, 39, 7, 26, 35, 28, 4, 14}
    # target_user_set_db2 = {11, 31, 17, 39, 7, 26, 35, 28, 4, 14,
    #                         32,19,38,24,12,18,23,21,30,3}
    # templist = list(range(1,41))
    # target_user_set_db2 = set(templist)
    target_user_set_db3 = {1, 2, 3, 4, 5, 6, 9, 10, 11}
    X = []
    y = []
    rep = []
    if exp_setup == None: # ninapro-db2-c50 is NOT implemented here yet.
        if filename[:11] == 'ninapro-db3':
            n_subject = 11
        else:
            n_subject = 40
        if filename[-3:] == 'c10':
            for subject in range(n_subject):
                data = loadmat(full_path + 'S' + str(subject + 1) + '_c10_200Hz.mat')
                X.extend(data['emg'])
                y.extend(data['restimulus'][0])
                rep.extend(data['rerepetition'][0])
        else:
            for subject in range(n_subject):
                data = loadmat(full_path + 'S' + str(subject + 1) + '_all_200Hz.mat')
                X.extend(data['emg'])
                y.extend(data['restimulus'])
                rep.extend(data['rerepetition'])

    elif exp_setup == 'per-subject': # ninapro-db2-c50 is ONLY implemented here yet.
        if 'c50' in filename:
            data = loadmat(full_path + 'S' + str(subject_idx) + '_2000Hz_3.mat')
            X.extend(data['emg'])
            y.extend(data['restimulus'].T[0])
            rep.extend(data['rerepetition'].T[0])
        elif 'c10' in filename:
            data = loadmat(full_path + 'S' + str(subject_idx) + '_c10_200Hz.mat')
            X.extend(data['emg'])
            y.extend(data['restimulus'][0])
            rep.extend(data['rerepetition'][0])
        else:
            data = loadmat(full_path + 'S' + str(subject_idx) + '_all_200Hz.mat')
            X.extend(data['emg'])
            y.extend(data['restimulus'])
            rep.extend(data['rerepetition'])

    elif exp_setup == 'leave-one-user-out': # ninapro-db2-c50 is NOT implemented here yet.
        if filename[:11] == 'ninapro-db3':
            n_subject = 11
            target_user_set = target_user_set_db3
        else:
            n_subject = 40
            target_user_set = target_user_set_db2
        X_eval = []
        X_test = []
        y_eval = []
        y_test = []
        if 'c10' in filename:
            for subject in range(1, n_subject + 1):
                if subject in target_user_set:
                    data = loadmat(full_path + 'S' + str(subject) + '_c10_200Hz.mat')
                    if subject == subject_idx:  # for test set
                        X_test.extend(data['emg'])
                        y_test.extend(data['restimulus'][0])
                    elif subject == 11:  # for eval set
                        X_eval.extend(data['emg'])
                        y_eval.extend(data['restimulus'][0])
                    else:
                        X.extend(data['emg'])
                        y.extend(data['restimulus'][0])
        else:
            for subject in range(1, n_subject + 1):
                if subject in target_user_set:
                    data = loadmat(full_path + 'S' + str(subject) + '_all_200Hz.mat')
                    if subject == subject_idx:  # for test set
                        X_test.extend(data['emg'])
                        y_test.extend(data['restimulus'][0])
                    elif subject == 11:  # for eval set
                        X_eval.extend(data['emg'])
                        y_eval.extend(data['restimulus'][0])
                    else:
                        X.extend(data['emg'])
                        y.extend(data['restimulus'][0])

    if exp_setup == 'leave-one-user-out':
        X = np.array(X)
        y = np.array(y)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        X_eval = np.array(X_eval)
        y_eval = np.array(y_eval)
        for i in range(len(y)):
            y[i] = target_gest_dict[y[i]]
        for i in range(len(y_test)):
            y_test[i] = target_gest_dict[y_test[i]]
        for i in range(len(y_eval)):
            y_eval[i] = target_gest_dict[y_eval[i]]
        data_raw = {'train_data': X, 'train_labels': y,
                    'val_data': X_eval, 'val_labels': y_eval,
                    'test_data': X_test, 'test_labels': y_test
                    }
    else:
        X = np.array(X)
        y = np.array(y)
        train_idx = []
        val_idx = []
        test_idx = []
        for i in range(len(rep)):
            if y[i] in target_gest_dict:
                if exp_setup == 'per-subject':  # one model per user
                    if rep[i] == 5:
                        test_idx.append(i)
                    elif rep[i] == 3:
                        val_idx.append(i)
                    else:
                        train_idx.append(i)
                else:  # one model for all users
                    if rep[i] == 2 or rep[i] == 5:
                        test_idx.append(i)
                    elif rep[i] == 3:
                        val_idx.append(i)
                    else:
                        train_idx.append(i)
                y[i] = target_gest_dict[y[i]]

        data_raw = {'train_data': X[train_idx, :], 'train_labels': y[train_idx],
                    'val_data': X[val_idx, :], 'val_labels': y[val_idx],
                    'test_data': X[test_idx, :], 'test_labels': y[test_idx]
                    }

    for dset in ['train', 'val', 'test']:
        print('The shape of the {} dataset is {} x {}, and the labels is {}'.format(dset, len(data_raw[dset + '_data']),
                                                                                    len(data_raw[dset + '_data'][0]),
                                                                                    len(data_raw[dset + '_labels'])))
        data_raw[dset + '_data'] = data_raw[dset + '_data'].astype(np.float32)
        data_raw[dset + '_labels'] = data_raw[dset + '_labels'].astype(np.uint8)

    return data_raw

def load_dataset_ninapro_db6_auth(full_path, filename):
    pass

def load_dataset_ninapro_auth(full_path, filename):
    # target_user_set_db2 = {11, 31, 17, 39, 7, 26, 35, 28, 4, 14,
    #                         32,19,38,24,12,18,23,21,30,3}
    # 11th user is omitted here since 11th user has the most amount of samples, thus used in GR.
    # here in Authentication, we use top 10 most-sampled users excluding 11th user.
    target_user_set_db2 = {31, 17, 39, 7, 26, 35, 28, 4, 14, 32}
    target_user_set_db3 = {1, 2, 3, 4, 5, 6, 9, 10, 11}
    target_gest_dict = {
        13: 0, 14: 1, 12: 2, 11: 3, 5: 4, 33: 5, 34: 6, 19: 7, 32: 8, 21: 9
    }
    X = []
    y = []
    rep = []
    gest = []

    n_subject = 40 if 'db2' in filename else 11
    target_user_set = target_user_set_db2 if 'db2' in filename else target_user_set_db3

    # DATA LOADING
    n_included_subj = 0
    for subject in range(n_subject):
        if (subject + 1) in target_user_set:
            if 'db2-c50' in filename: # in case of static features
                data = loadmat(full_path + 'S' + str(subject + 1) + '_2000Hz_3.mat')
                rep.extend(data['rerepetition'].T[0])
                gest.extend(data['restimulus'].T[0])
            elif 'c10' in filename:
                data = loadmat(full_path + 'S' + str(subject + 1) + '_c10_200Hz.mat')
                rep.extend(data['rerepetition'][0])
                gest.extend(data['restimulus'][0])
            else: # in case of faw features
                data = loadmat(full_path + 'S' + str(subject + 1) + '_all_200Hz.mat')
                rep.extend(data['rerepetition'])
                gest.extend(data['restimulus'])
            X.extend(data['emg'])
            y += [n_included_subj] * data['emg'].shape[0]
            n_included_subj += 1

    # Separate data into train, val, test sets according to repetition num
    X = np.array(X)
    y = np.array(y)
    train_idx = []
    val_idx = []
    test_idx = []
    for i in range(len(rep)):
        if gest[i] in target_gest_dict:
            if rep[i] == 5:
                test_idx.append(i)
            elif rep[i] == 3:
                val_idx.append(i)
            else:
                train_idx.append(i)

    data_raw = {'train_data': X[train_idx, :], 'train_labels': y[train_idx],
                'val_data': X[val_idx, :], 'val_labels': y[val_idx],
                'test_data': X[test_idx, :], 'test_labels': y[test_idx]
                }

    for dset in ['train', 'val', 'test']:
        print('The shape of the {} dataset is {} x {}, and the labels is {}'.format(
            dset, len(data_raw[dset + '_data']),
            len(data_raw[dset + '_data'][0]),
            len(data_raw[dset + '_labels'])))
        data_raw[dset + '_data'] = data_raw[dset + '_data'].astype(np.float32)
        data_raw[dset + '_labels'] = data_raw[dset + '_labels'].astype(np.uint8)

    return data_raw


def load_dataset(filename):
    """
    Loading the .mat file and creating a dictionary based on the phase
    :param filename: name of the .mat file
    :return: dictionary containing the sensory data
    """
    # Load the data from the .mat file
    data = loadmat(filename)

    # Putting together the data into a dictionary for easy retrieval
    data_raw = {'train_data': data['X_train'], 'train_labels': np.transpose(data['y_train']),
                'val_data': data['X_valid'],
                'val_labels': np.transpose(data['y_valid']), 'test_data': data['X_test'],
                'test_labels': np.transpose(data['y_test'])}

    # Setting the variable types for the data and labels
    for dset in ['train', 'val', 'test']:
        print('The shape of the {} dataset is {}, and the labels is {}'.format(dset, len(data_raw[dset + '_data'][0]),
                                                                               len(data_raw[dset + '_labels']) ))
        data_raw[dset + '_data'] = data_raw[dset + '_data'].astype(np.float32)
        data_raw[dset + '_labels'] = data_raw[dset + '_labels'].astype(np.uint8)

    return data_raw


class MyDataset(torch.utils.data.Dataset):
    """docstring for Custom Dataset"""
    def __init__(self, X, y, transform = None, target_transform = None):
        super(MyDataset, self).__init__()
        self.X, self.y = X, y
        self.target_transform = target_transform
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X, y = self.X[index], self.y[index]
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)
        return X, y

    def get_data_dim(self):
        return list(self.X.size())

