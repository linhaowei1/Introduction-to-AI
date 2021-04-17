from PreActResNet18 import PreActResNet18
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import pdb
from tensorboardX import SummaryWriter
import random
from utils import get_args


if __name__ == '__main__':
    args = get_args()
    #writer = SummaryWriter(log_dir=args.log_dir)
    device = args.cuda
    with torch.cuda.device(int(device[-1])):
        # 设置随机种子
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)

        model = PreActResNet18()
        model.to(device)

        model.load_state_dict(torch.load(
            './CIFAR10_PreActResNet18.checkpoint', map_location=device))

        data_train = datasets.CIFAR10(
            root='/home/linhw/myproject/data/CIFAR10',
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            download=True
        )
        data_test = datasets.CIFAR10(
            root='/home/linhw/myproject/data/CIFAR10',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            download=True
        )

        number_worker = min(
            [os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])

        train_loader = torch.utils.data.DataLoader(data_train,
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=number_worker)
        test_loader = torch.utils.data.DataLoader(data_test,
                                                  batch_size=args.batch_size, shuffle=True,
                                                  num_workers=number_worker)
        model.eval()
        acc = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for test_data in test_bar:
                test_images, test_labels = test_data
                outp = model(test_images.to(device))
                pred = torch.max(outp, dim=1)[1]
                acc += torch.eq(pred, test_labels.to(device)).sum().item()

        acc /= test_num
        print("test acc = {:.5f}".format(acc))
