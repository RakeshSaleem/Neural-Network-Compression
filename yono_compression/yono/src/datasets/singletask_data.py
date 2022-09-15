
from src.datasets.utils import *


def get_singletask_data(args, mode=None, transform=None, target_transform=None,
                        subject_idx=None,
                        exp_setup=None,
                        test_fold_l=[10]):

    args.cls_type = args.model_arch

    X_train, y_train, X_test, y_test = load_and_segment_data(
        name=args.dataset, config=args.config, subject_idx=subject_idx, exp_setup=exp_setup,
        test_fold_l=test_fold_l, args=args)

    X_train = reshape(X_train)
    X_test = reshape(X_test)

    X_train, y_train = convert_2_torch(X_train, y_train)
    X_test, y_test = convert_2_torch(X_test, y_test)

    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)

    return train_dataset, test_dataset