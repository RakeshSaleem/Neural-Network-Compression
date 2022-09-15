# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'mnist': {'in_channels':1, 'features': 28, 'seq': 28, 'classes': 10},
    'fashion-mnist': {'in_channels':1, 'features': 28, 'seq': 28, 'classes': 10},
    'cifar10': {'in_channels':3,'features': 32, 'seq': 32, 'classes': 10},
    'cifar10-vw': {'in_channels':3,'features': 32, 'seq': 32, 'classes': 10},
    'svhn': {'in_channels':3,'features': 32, 'seq': 32, 'classes': 10},
    'svhn-vw': {'in_channels':3,'features': 32, 'seq': 32, 'classes': 10},
    'gtsrb': {'in_channels':3,'features': 32, 'seq': 32, 'classes': 43},
    'gtsrb-vw': {'in_channels':1,'features': 32, 'seq': 32, 'classes': 43},
    'cifar100': {'in_channels':3,'features': 32, 'seq': 32, 'classes': 100},
    'stl10': {'in_channels':3,'features': 96, 'seq': 96, 'classes': 10},
    'imagenet': {'in_channels':3,'features': 224, 'seq': 224, 'classes': 1000},
    'opportunity': {'in_channels':1,'features': 113, 'seq': 24, 'classes': 17}, # w/o null class
    'hhar-raw': {'in_channels':1,'features': 6, 'seq': 50, 'classes': 6}, # no null class
    'hhar-noaug': {'in_channels':1,'features': 120, 'seq': 20, 'classes': 6}, # no null class
    'hhar-aug': {'in_channels':1,'features': 120, 'seq': 20, 'classes': 6}, # no null class
    'opp_thomas': {'in_channels':1,'features': 77, 'seq': 30, 'classes': 18}, # w/ null class
    'pamap2': {'in_channels':1,'features': 52, 'seq': 33, 'classes': 12}, # w/o null class
    'skoda': {'in_channels':1,'features': 60, 'seq': 33, 'classes': 10}, # w/ null class
    'usc-had': {'in_channels':1,'features': 6, 'seq': 33, 'classes': 12},
    'ninapro-db2-c10': {'in_channels':1,'features': 12, 'seq': 40, 'classes': 10}, # w/o null class
    'ninapro-db2-c10-seq80': {'in_channels':1,'features': 12, 'seq': 80, 'classes': 10}, # w/o null class
    'ninapro-db2-c10-seq100': {'in_channels':1,'features': 12, 'seq': 100, 'classes': 10}, # w/o null class
    'ninapro-db2-c50': {'in_channels':1,'features': 372, 'seq': 1, 'classes': 10}, # w/o null class
    'ninapro-db3-c10': {'in_channels':1,'features': 12, 'seq': 40, 'classes': 10}, # w/o null class
    'ninapro-db3-c10-seq80': {'in_channels':1,'features': 12, 'seq': 80, 'classes': 10}, # w/o null class
    'ninapro-db3-c10-seq100': {'in_channels':1,'features': 12, 'seq': 100, 'classes': 10}, # w/o null class
    'ninapro-db6-c7-seq50': {'in_channels':1,'features': 8, 'seq': 50, 'classes': 7}, # w/o null class
    'ninapro-db6-c7-seq100': {'in_channels':1,'features': 8, 'seq': 100, 'classes': 7}, # w/o null class
    'ninapro-db6-c7-seq80': {'in_channels':1,'features': 8, 'seq': 80, 'classes': 7}, # w/o null class
    'ninapro-db6-c7-seq40': {'in_channels':1,'features': 8, 'seq': 40, 'classes': 7}, # w/o null class
    'ninapro-db2-c10-auth': {'in_channels':1,'features': 12, 'seq': 40, 'classes': 10}, # authentication
    'ninapro-db2-c50-auth': {'in_channels':1,'features': 372, 'seq': 1, 'classes': 10}, # authentication
    'ninapro-db3-c10-auth': {'in_channels':1,'features': 12, 'seq': 40, 'classes': 10}, # authentication
    'ninapro-db6-c10-auth': {'in_channels':1,'features': 8, 'seq': 50, 'classes': 10}, # authentication
    'emotion-14-frames10': {'in_channels':1,'features': 24, 'seq': 50, 'classes': 14},
    'emotion-14-frames20': {'in_channels':1,'features': 24, 'seq': 25, 'classes': 14},
    'emotion-14-frames25': {'in_channels':1,'features': 24, 'seq': 20, 'classes': 14},
    'emotion-14-frames50': {'in_channels':1,'features': 24, 'seq': 10, 'classes': 14},
    'emotion-5-frames10': {'in_channels':1,'features': 24, 'seq': 50, 'classes': 5},
    'emotion-5-frames20': {'in_channels':1,'features': 24, 'seq': 25, 'classes': 5},
    'emotion-5-frames25': {'in_channels':1,'features': 24, 'seq': 20, 'classes': 5},
    'emotion-5-frames50': {'in_channels':1,'features': 24, 'seq': 10, 'classes': 5},
    'urbansound8k': {'in_channels':1,'features': 128, 'seq': 128, 'classes': 10},
    'urbansound8k-augment': {'in_channels':1,'features': 128, 'seq': 42, 'classes': 10},
    'urbansound8k-LMCST': {'in_channels':1,'features': 85, 'seq': 43, 'classes': 10},
    'urbansound8k-LMCST-long': {'in_channels':1,'features': 85, 'seq': 128, 'classes': 10},
    'urbansound8k-LMCST-long-1s': {'in_channels':1,'features': 85, 'seq': 41, 'classes': 10},
    'urbansound8k-LMCST-long-randfold': {'in_channels':1,'features': 85, 'seq': 128, 'classes': 10},
    'urbansound8k-LMCST-1s-45f-randfold': {'in_channels':1,'features': 45, 'seq': 41, 'classes': 10},
    'urbansound8k-LMCST-1s-fullhop-45f-randfold': {'in_channels':1,'features': 45, 'seq': 21, 'classes': 10},
    'urbansound8k-LMCST-long-1s-randfold': {'in_channels':1,'features': 85, 'seq': 41, 'classes': 10},
    'urbansound8k-LMCST-long-shuffle': {'in_channels':1,'features': 85, 'seq': 128, 'classes': 10},
    'urbansound8k-LMCST-long-1s-shuffle': {'in_channels':1,'features': 85, 'seq': 41, 'classes': 10},
    'urbansound8k-augment-vote': {'in_channels':1,'features': 128, 'seq': 42, 'classes': 10},
    'urbansound8k-LMCST-vote': {'in_channels':1,'features': 85, 'seq': 43, 'classes': 10},
    'urbansound8k-LMCST-vote-randfold': {'in_channels':1,'features': 85, 'seq': 43, 'classes': 10},
    'urbansound8k-LMCST-vote-shuffle': {'in_channels':1,'features': 85, 'seq': 43, 'classes': 10},
    'urbansound8k-LMCST-4s-long': {'in_channels':1,'features': 85, 'seq': 170, 'classes': 10},
    'urbansound8k-LMCST-4s-long-randfold': {'in_channels':1,'features': 85, 'seq': 170, 'classes': 10},
    'urbansound8k-LMCST-4s-long-shuffle': {'in_channels':1,'features': 85, 'seq': 170, 'classes': 10},
    'urbansound8k-LMCST-4s-vote': {'in_channels':1,'features': 85, 'seq': 41, 'classes': 10},
    'urbansound8k-LMCST-4s-vote-randfold': {'in_channels':1,'features': 85, 'seq': 41, 'classes': 10},
    'urbansound8k-LMCST-4s-vote-shuffle': {'in_channels':1,'features': 85, 'seq': 41, 'classes': 10},
    'urbansound8k-LM': {'in_channels':1,'features': 60, 'seq': 128, 'classes': 10},
    'urbansound8k-LM-randfold': {'in_channels':1,'features': 60, 'seq': 128, 'classes': 10},
    'urbansound8k-LM-shuffle': {'in_channels':1,'features': 60, 'seq': 128, 'classes': 10},
    'gscv2-mfcc10': {'in_channels':1,'features': 10, 'seq': 49, 'classes': 12},
    'gscv2-mfcc40': {'in_channels':1,'features': 40, 'seq': 49, 'classes': 12},
    'gscv2-mfcc10-c35': {'in_channels':1,'features': 10, 'seq': 49, 'classes': 35},
    'gscv2-mfcc40-c35': {'in_channels':1,'features': 40, 'seq': 49, 'classes': 35},
    'gscv2-vw-c35': {'in_channels':1,'features': 61, 'seq': 13, 'classes': 35},
    'voxceleb1-f32-t32': {'in_channels':1,'features': 32, 'seq': 32, 'classes': 1251}
}