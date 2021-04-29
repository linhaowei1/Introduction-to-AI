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
from model import resnet18, LeNet


def fgsm(eps=0.031, criterion=F.nll_loss, bool_rand_noise=False):
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

        outp_1 = sub_model(image)
        pred_1 = torch.max(outp_1, dim=1)[1]
        mask = (pred_1 == label).to(device)
        total += mask.sum().item()

        # 是否加噪声
        if bool_rand_noise:
            random_noise = torch.FloatTensor(
                *image.shape).uniform_(-eps, eps).to(device)
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

        outp_2 = sub_model(perturbed_image)
        pred_2 = torch.max(outp_2, dim=1)[1]
        cor = (pred_2 == label).to(device) * mask
        total_correct += cor.sum().item()

    print('[FGSM] succ_rate: %.5f' % ((total-total_correct)/total))
    suc = (total-total_correct)/total
    with open('./black_results.txt', 'a+') as f:
        f.write(f'dataset={args.dataset}\tattack={args.attack}\tnoise={args.noise}\tsucc_rate={suc}\tloss={args.loss}\n')
    f.close()


def pgd(eps=0.031, step_size=0.003, epoch=10, criterion=F.cross_entropy, bool_rand_noise=False):
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
        outp_1 = sub_model(image)
        pred_1 = torch.max(outp_1, dim=1)[1]
        # mask : 掩码，考虑判断正确的样本
        mask = (pred_1 == label).to(device)
        # total : 分类正确的个数
        total += mask.sum().item()
        # 是否加噪声
        if bool_rand_noise:
            random_noise = torch.FloatTensor(
                *image.shape).uniform_(-eps, eps).to(device)
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

        outp_2 = sub_model(perturbed_image)
        pred_2 = torch.max(outp_2, dim=1)[1]
        cor = (pred_2 == label).to(device) * mask
        total_correct += cor.sum().item()

    print('[PGD] succ_rate: %.5f' % ((total-total_correct)/total))

    suc = (total-total_correct)/total
    with open('./black_results.txt', 'a+') as f:
        f.write(f'dataset={args.dataset}\tattack={args.attack}\tnoise={args.noise}\tsucc_rate={suc}\tstepnumber={args.step_number}\tstepsize={args.stepsize}\tloss={args.loss}\n')
    f.close()


def mi_fgsm(eps=0.031, step_size=0.003, epoch=10, criterion=F.cross_entropy, bool_rand_noise=False, alpha=1.0):
    #time = 54s
    model.eval()
    sub_model.eval()
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
        outp_1 = sub_model(image)
        pred_1 = torch.max(outp_1, dim=1)[1]
        # mask : 掩码，考虑判断正确的样本
        mask = (pred_1 == label).to(device)
        # total : 分类正确的个数
        total += mask.sum().item()
        # 是否加噪声
        if bool_rand_noise:
            random_noise = torch.FloatTensor(
                *image.shape).uniform_(-eps, eps).to(device)
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
            grad_norm = torch.norm(image_grad.reshape(-1), 1)
            image_grad = image_grad/grad_norm + alpha*momentum
            momentum = image_grad

            perturbed_image = perturbed_image + step_size * image_grad.sign()
            eta = torch.clamp(perturbed_image.data - image.data, -eps, eps)
            perturbed_image = image + eta
            perturbed_image = torch.clamp(perturbed_image, 0, 1)

        ## 可视化图像
        writer.add_image('image', image[0], batch)
        writer.add_image('perturb_image', perturbed_image[0], batch)

        outp_2 = sub_model(perturbed_image)
        pred_2 = torch.max(outp_2, dim=1)[1]
        cor = (pred_2 == label).to(device) * mask
        total_correct += cor.sum().item()

    print('[Mi_FGSM] succ_rate: %.5f/%.5f' % ((total-total_correct)/total, total))
    suc = (total-total_correct)/total
    with open('./black_results.txt', 'a+') as f:
        f.write(f'dataset={args.dataset}\tattack={args.attack}\tnoise={args.noise}\tsucc_rate={suc}\tstepnumber={args.step_number}\tstepsize={args.stepsize}\tloss={args.loss}\n')
    f.close()

def lr_schedule_func(epoch):
    if epoch < 135:
        return 0.1
    elif epoch < 200:
        return 0.01
    else:
        return 0.001 

def pretrain():
    best_acc = 0.0
    for epoch in range(args.epochs):
        sub_model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader)
        for batch, data in enumerate(train_bar):
            images, labels = data
            logits = sub_model(images.to(device))
            optimizer.zero_grad()
            loss = loss_func(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.5f}".format(
                epoch+1, args.epochs, total_loss / (len(train_bar) * (batch + 1)))

        writer.add_scalar('train/loss', total_loss / len(data_train), epoch)

        sub_model.eval()
        acc = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for test_data in test_bar:
                test_images, test_labels = test_data
                outp = sub_model(test_images.to(device))
                pred = torch.max(outp, dim=1)[1]
                acc += torch.eq(pred,
                                test_labels.to(device)).sum().item()

                test_bar.desc = "test epoch[{}/{}]".format(
                    epoch + 1, args.epochs)

        acc /= len(data_test)
        if acc > best_acc:
            torch.save(sub_model.state_dict(), f'{args.dataset}_pretrain.pth')
            best_acc = acc
        writer.add_scalar('test/acc', acc, epoch)
        #scheduler.step()
        print('[epoch %d] train_loss: %.5f  test_accuracy: %.5f' %
              (epoch + 1, total_loss / len(data_train), acc))

    print("Finished Training!")


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
            sub_model = resnet18(pretrained=True).cuda()
            # 修改channel数（1000分类->10分类）
            in_channel = sub_model.fc.in_features
            sub_model.fc = nn.Linear(in_channel, 10)
            sub_model = sub_model.to(device)
            loss_func = nn.CrossEntropyLoss()
            loss_func.to(device)
            optimizer = optim.Adam(sub_model.parameters(), lr=1e-4) 
            if not args.pretrain:
                sub_model.load_state_dict(torch.load('./cifar_pretrain.pth', map_location=device))

            model.load_state_dict(torch.load(
                './CIFAR10_PreActResNet18.pth', map_location=device)['state_dict'])
            data_train = datasets.CIFAR10(
                root='/home/linhw/myproject/data/CIFAR10',
                train=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                ]),
                download=True
            )
            data_test = datasets.CIFAR10(
                root='/home/linhw/myproject/data/CIFAR10',
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                ]),
                download=True
            )
        elif args.dataset == 'MNIST':
            sub_model = LeNet().cuda()
            sub_model = sub_model.to(device)
            loss_func = nn.CrossEntropyLoss()
            loss_func.to(device)
            optimizer = optim.SGD(sub_model.parameters(), lr=args.lr,momentum=0.9, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
            if not args.pretrain:
                sub_model.load_state_dict(torch.load('./MNIST_pretrain.pth', map_location=device))
            model.load_state_dict(torch.load(
                './MNIST_small_cnn.pth', map_location=device)['state_dict'])
            data_train = datasets.MNIST(
                root='/home/linhw/myproject/data/MNIST',
                train=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                ]),
                download=True
            )
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

        train_loader = torch.utils.data.DataLoader(data_train,
                                                  batch_size=args.batch_size, shuffle=True,
                                                  num_workers=number_worker)
        test_loader = torch.utils.data.DataLoader(data_test,
                                                  batch_size=args.batch_size, shuffle=True,
                                                  num_workers=number_worker)
        if args.pretrain:
            pretrain()
        else:
            model.eval()
            sub_model.eval()

            if args.attack == 'fgsm':
                fgsm(eps=args.epsilon)
            elif args.attack == 'mi_fgsm':
                mi_fgsm(eps=args.epsilon, epoch=args.step_number,
                        step_size=args.stepsize)
            elif args.attack == 'pgd':
                pgd(eps=args.epsilon, epoch=args.step_number, step_size=args.stepsize)
            else:
                raise NotImplementedError
