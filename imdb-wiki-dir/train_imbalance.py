import argparse
from resnet18 import resent18_regression
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd
from datasets import IMDBWIKI

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='imdb_wiki', choices=['imdb_wiki', 'agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='./data', help='data directory')

parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()

args.start_epoch, args.best_loss = 0, 1e5

def get_dataset():
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    #
    # limit the age range from 26 to 28
    # only limit the 26,27,28 (6562(6262), 6414(6114), 6742(6442) samples respectively(train))
    #
    df_train_26, df_val_26, df_test_26 = df_train[df_train['age'] == 26], df_val[df_val['age'] == 26], df_test[df_test['age'] == 26]
    df_train_27, df_val_27, df_test_27 = df_train[df_train['age'] == 27], df_val[df_val['age'] == 27], df_test[df_test['age'] == 27]
    df_train_28, df_val_28, df_test_28 = df_train[df_train['age'] == 28], df_val[df_val['age'] == 28], df_test[df_test['age'] == 28]
    # train
    # use 1000 27 and 1000 28 to train
    df_train_26 = shuffle(df_train_26)
    df_train_27 = shuffle(df_train_27)
    df_train_28 = shuffle(df_train_28)
    df_train = pd.concat([df_train_26[:args.train_number], df_train_27[:1000], df_train_28[:1000]])
    df_train = shuffle(df_train)
    # test
    df_test = pd.concat([df_test_27, df_test_28])
    df_test = shuffle(df_test)
    # val
    df_val = pd.concat([df_val_27, df_val_28])
    df_val = shuffle(df_val)
    #    

    train_labels = df_train['age']

    train_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_train, img_size=args.img_size, split='train',
                             reweight=args.reweight, lds=args.lds, lds_kernel=args.lds_kernel, lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)
    val_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_val, img_size=args.img_size, split='val')
    test_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_test, img_size=args.img_size, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    return train_loader, test_loader, val_loader

def get_model():
    model = resent18_regression()
    return model

def train():
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")
    train_loader, test_loader, _ = get_dataset()
    model = get_model()
    criterion_mse = nn.MSELoss()
    opt = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)
    # train
    model.train()
    for epoch in range(100):
        for idx, (inputs, targets, weights) in enumerate(train_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = criterion_mse(outputs, targets)
            loss.backwards()
            opt.step()
    return model , loss
