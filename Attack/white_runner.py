from PreActResNet18 import PreActResNet18
from small_cnn import SmallCNN
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

def fgsm(eps=0.031, criterion=F.nll_loss, bool_rand_noise=False):
    # fgsm运行时间：5秒
    # loss=ce，bool_noise=True: 0.93103
    # loss=ce, bool_noise=False: 0.86524
    # loss=nll, bool_noise=True: 0.93919
    # loss=nll, bool_noise=False: 0.89384
    # loss=nll, noise=normal(0,eps) : 0.94215 
    # loss=nll, noise=normal(0,eps**0.5) : 0.38532 noise(这个太大了)
    # loss=nll, noise=normal(0,eps*2) : 0.91525 noise(这个有点大，还是eps比较好)
    # loss=nll, noise=normal(0,eps*0.5) : 0.92425 noise(这个比较小了，还是eps比较好)
    # loss=nll, noise=normal(0,eps**2) : 0.89363 noise(这个太小了，和不加noise没啥差别)
    if args.noise:
        bool_rand_noise = True

    if args.loss == 'ce':
        criterion = F.cross_entropy
    elif args.loss == 'nll':
        criterion = F.nll_loss
    else:
        raise NotImplementedError
    total_correct = 0
    total = 0
    adv_samples = []
    test_bar = tqdm(test_loader)
    for batch, (image, label) in enumerate(test_bar):
        image, label = image.to(device), label.to(device)
        outp_1 = model(image)
        pred_1 = torch.max(outp_1, dim=1)[1]
        # mask : 掩码，考虑判断正确的样本
        mask = (pred_1 == label).to(device)
        # total : 分类正确的个数
        total += mask.sum().item()
        # 是否加噪声
        if bool_rand_noise:
            random_noise = torch.FloatTensor(*image.shape).uniform_(-eps, eps).to(device)
        else:
            #如果不加噪声，噪声就是0向量
            random_noise = torch.zeros_like(image).float().to(device)
        perturbed_image = image + random_noise
        perturbed_image.requires_grad = True
        loss = criterion(model(perturbed_image), label)
        model.zero_grad()
        loss.backward()
        image_grad = perturbed_image.grad
        perturbed_image = perturbed_image + image_grad.sign()
        eta = torch.clamp(perturbed_image.data - image.data, -eps, eps)
        perturbed_image = image + eta
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        ## 可视化图像
        writer.add_image('image', image[0], batch)
        writer.add_image('perturb_image', perturbed_image[0], batch)

        outp_2 = model(perturbed_image)
        pred_2 = torch.max(outp_2, dim=1)[1]
        cor = (pred_2 == label).to(device) * mask
        total_correct += cor.sum().item()
    
    print('[FGSM] succ_rate: %.5f' % ((total-total_correct)/total))
    suc = (total-total_correct)/total
    with open('./results.txt', 'a+') as f:
        f.write(f'dataset={args.dataset}\tattack={args.attack}\tnoise={args.noise}\tsucc_rate={suc}\tloss={args.loss}\n')
    f.close()

def pgd(eps=0.031, step_size=0.003, epoch=10, criterion=F.cross_entropy, bool_rand_noise=False):
    # pgd运行时间：35秒
    # loss使用ce，加noise：0.99989
    # loss使用nll，加noise：0.99979
    # loss使用ce，不加noise：1.00000
    # loss使用nll，不加noise：0.99989
    if args.noise:
        bool_rand_noise = True
    total_correct = 0
    total = 0
    adv_samples = []
    if args.loss == 'ce':
        criterion = F.cross_entropy
    elif args.loss == 'nll':
        criterion = F.nll_loss
    else:
        raise NotImplementedError
    test_bar = tqdm(test_loader)
    for batch, (image, label) in enumerate(test_bar):
        image, label = image.to(device), label.to(device)
        outp_1 = model(image)
        pred_1 = torch.max(outp_1, dim=1)[1]
        # mask : 掩码，考虑判断正确的样本
        mask = (pred_1 == label).to(device)
        # total : 分类正确的个数
        total += mask.sum().item()
        # 是否加噪声
        if bool_rand_noise:
            random_noise = torch.FloatTensor(*image.shape).uniform_(-eps, eps).to(device)
        else:
            #如果不加噪声，噪声就是0向量
            random_noise = torch.zeros_like(image).float().to(device)
        perturbed_image = image + random_noise
        for _ in range(epoch):
            perturbed_image.requires_grad = True
            loss = criterion(model(perturbed_image), label)
            model.zero_grad()
            loss.backward()
            image_grad = perturbed_image.grad
            perturbed_image = perturbed_image + step_size * image_grad.sign()
            eta = torch.clamp(perturbed_image.data - image.data, -eps, eps)
            perturbed_image = image + eta
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
       
        ## 可视化图像
        writer.add_image('image', image[0], batch)
        writer.add_image('perturb_image', perturbed_image[0], batch)

        outp_2 = model(perturbed_image)
        pred_2 = torch.max(outp_2, dim=1)[1]
        cor = (pred_2 == label).to(device) * mask
        total_correct += cor.sum().item()
    
    print('[PGD] succ_rate: %.5f' % ((total-total_correct)/total))

    suc = (total-total_correct)/total
    with open('./results.txt', 'a+') as f:
        f.write(f'dataset={args.dataset}\tattack={args.attack}\tnoise={args.noise}\tsucc_rate={suc}\tstepnumber={args.step_number}\tstepsize={args.stepsize}\tloss={args.loss}\n')
    f.close()

def mi_fgsm(eps=0.031, step_size=0.003, epoch=10, criterion=F.cross_entropy, bool_rand_noise=False, alpha=1.0):
    #time = 54s
    if args.noise:
        bool_rand_noise = True
    total_correct = 0
    total = 0
    adv_samples = []
    if args.loss == 'ce':
        criterion = F.cross_entropy
    elif args.loss == 'nll':
        criterion = F.nll_loss
    else:
        raise NotImplementedError
    test_bar = tqdm(test_loader)
    for batch, (image, label) in enumerate(test_bar):
        image, label = image.to(device), label.to(device)
        momentum = torch.zeros_like(image).to(device)
        outp_1 = model(image)
        pred_1 = torch.max(outp_1, dim=1)[1]
        # mask : 掩码，考虑判断正确的样本
        mask = (pred_1 == label).to(device)
        # total : 分类正确的个数
        total += mask.sum().item()
        # 是否加噪声
        if bool_rand_noise:
            random_noise = torch.FloatTensor(*image.shape).uniform_(-eps, eps).to(device)
        else:
            #如果不加噪声，噪声就是0向量
            random_noise = torch.zeros_like(image).float().to(device)
        perturbed_image = image + random_noise
        for _ in range(epoch):
            perturbed_image.requires_grad = True
            loss = criterion(model(perturbed_image), label)
            model.zero_grad()
            loss.backward()
            image_grad = perturbed_image.grad
            grad_norm = torch.norm(image_grad.reshape(-1),1)
            image_grad = image_grad/grad_norm + alpha*momentum
            momentum = image_grad
            
            perturbed_image = perturbed_image + step_size * image_grad.sign()
            eta = torch.clamp(perturbed_image.data - image.data, -eps, eps)
            perturbed_image = image + eta
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
       
        ## 可视化图像
        writer.add_image('image', image[0], batch)
        writer.add_image('perturb_image', perturbed_image[0], batch)

        outp_2 = model(perturbed_image)
        pred_2 = torch.max(outp_2, dim=1)[1]
        cor = (pred_2 == label).to(device) * mask
        total_correct += cor.sum().item()

    print('[Mi_FGSM] succ_rate: %.5f' % ((total-total_correct)/total))
    suc = (total-total_correct)/total
    with open('./results.txt', 'a+') as f:
        f.write(f'dataset={args.dataset}\tattack={args.attack}\tnoise={args.noise}\tsucc_rate={suc}\tstepnumber={args.step_number}\tstepsize={args.stepsize}\tloss={args.loss}\n')
    f.close()

if __name__ == '__main__':
    args = get_args()
    writer = SummaryWriter(log_dir=args.log_dir)
    device = args.cuda
    with torch.cuda.device(int(device[-1])):
        # 设置随机种子
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)

        if args.model == 'PreActResNet18':
            model = PreActResNet18()
        elif args.model == 'small_cnn':
            model = SmallCNN()
        else:
            raise NotImplementedError

        model.to(device)

        if args.dataset == 'cifar':
            model.load_state_dict(torch.load('./CIFAR10_PreActResNet18.pth', map_location=device)['state_dict'])
            data_test = datasets.CIFAR10(
                root='/home/linhw/myproject/data/CIFAR10',
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                ]),
                download=True
            )
        elif args.dataset == 'MNIST':
            model.load_state_dict(torch.load('./MNIST_small_cnn.pth', map_location=device)['state_dict'])
            data_test = datasets.MNIST(
                root='/home/linhw/myproject/data/MNIST',
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                ]),
                download=True
            )

        number_worker = min(
            [os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])

        test_loader = torch.utils.data.DataLoader(data_test,
                                                  batch_size=args.batch_size, shuffle=True,
                                                  num_workers=number_worker)
        model.eval()
        if args.attack == 'fgsm':
            fgsm(eps=args.epsilon)
        elif args.attack == 'mi_fgsm':
            mi_fgsm(eps=args.epsilon, epoch=args.step_number, step_size=args.stepsize)
        elif args.attack == 'pgd':
            pgd(eps=args.epsilon, epoch=args.step_number, step_size=args.stepsize)
        else:
            raise NotImplementedError
        