import argparse
from mimetypes import init
from resnet18 import resent18_regression, resnet18_cls
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd
from datasets import IMDBWIKI

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='imdb_wiki', choices=['imdb_wiki', 'agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--workers', type=int, default=8, help='number of workers')
parser.add_argument('--img_size', type=int, default=224, help='image size used in training')
parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()

args.start_epoch, args.best_loss = 0, 1e5

def get_dataset(args, leave_out_train = False, leave_num = 11):
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    leave_list = [20,21,22,23,24,25,26,27,28,29,30]
    #
    # limit the age range from 26 to 28
    # only limit the 26,27,28 (6562(6262), 6414(6114), 6742(6442) samples respectively(train))
    #
    #if leave_three_train:
        #leave_list = [20,21,22,23,24,25,26,27,28,29,30]
    train_list = []
    test_list = []
    val_list = []
    for i in leave_num:
        age_is = leave_list[i]
        df_train_cur, df_val_cur, df_test_cur = df_train[df_train['age'] == \
            age_is], df_val[df_val['age'] == age_is], df_test[df_test['age'] == age_is]
        df_train_cur, df_val_cur, df_test_cur = shuffle(df_train_cur), shuffle(df_val_cur), shuffle(df_test_cur)
        if leave_out_train:
            if i == 0:
                df_train_cur = df_train_cur[:args.train_number]
                train_list.append(df_train_cur)
            else:
                train_list.append(df_train_cur[:1000])
                test_list.append(df_test_cur[:1000])
                val_list.append(df_val_cur[:1000])
        else:
            train_list.append(df_train_cur)
            test_list.append(df_test_cur)
            val_list.append(df_val_cur)
        ####
    df_train = pd.concat(train_list)
    df_test = pd.concat(test_list)
    df_val = pd.concat(val_list)
    '''
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
    '''    

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

def get_model(pattern = 'cls'):
    if pattern == 'reg' :
        model = resent18_regression()
    else:
        model = resnet18_cls()
    model = torch.nn.DataParallel(model).cuda()
    return model

def train_step(train_loader, pattern = 'cls', dual = False):
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")


    if dual:
        model_reg = get_model('cls')
        model_cls = get_model('reg')
        criterion_reg = nn.MSELoss()
        criterion_cls = nn.CELoss()
        opt_cls = optim.SGD(model_cls.parameters(), lr=args.lr, weight_decay=5e-4)
        opt_reg = optim.SGD(model_reg.parameters(), lr=args.lr, weight_decay=5e-4)
        model_reg.train()
        model_cls.train()
        for epoch in range(100):
            for idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                outputs_cls = model_cls(inputs)
                outputs_reg = model_reg(inputs)
                loss_cls = criterion_cls(outputs_cls, inputs)
                loss_reg = criterion_reg(outputs_reg, inputs)
                loss_cls.backwards()
                loss_reg.backwards()
                opt_cls.step()
                opt_reg.step()
    else:
        # train
        model = get_model(pattern)
        opt = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)
        if pattern == 'reg':
            criterion = nn.MSELoss()
        elif pattern == 'cls':
            criterion = nn.CELoss()
        else:
            print(" no training patterns definied ")
        model.train()
        for epoch in range(100):
            for idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backwards()
                opt.step()
    return model , loss

def test_step(model, test_loader, pattern = 'cls'):
    criterion_mse = nn.MSELoss()
    losses_all = []
    maxk = max(topk)
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            # if indices starts from 0
            #output_hat = torch.max(outputs, 1).indices + 1
            if pattern == 'cls':
                _, pred = outputs.topk(maxk, 1, True, True)
            else:
                pred = outputs
            loss = criterion_mse(pred, targets)
            losses_all.append(loss.item())
    return max(losses_all)


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_cls_loader, test_cls_loader, val_cls_loader = get_dataset()
    train_reg_loader, test_reg_loader, val_reg_loader = get_dataset()
    model_cls, _ = train_step(train_cls_loader, 'cls')
    model_reg, _ = train_step(train_reg_loader, 'reg')




