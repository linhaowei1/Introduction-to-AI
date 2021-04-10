import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import pdb
from tensorboardX import SummaryWriter
import random
from model import resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet101_2, wide_resnet50_2, resnext101_32x8d, resnext50_32x4d, resnet152
from utils import get_args

def train():
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader)
        for batch, data in enumerate(train_bar):
            images, labels = data
            logits = model(images.to(cuda_device))
            optimizer.zero_grad()
            loss = loss_func(logits, labels.to(cuda_device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.5f}".format(epoch+1, args.epochs, total_loss / (len(train_bar) * (batch + 1)))

        writer.add_scalar('train/loss', total_loss / train_num, epoch)
        
        if args.mode == 'adjust_parameters':
            model.eval()
            acc = 0.0
            with torch.no_grad():
                val_bar = tqdm(val_loader)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outp = model(val_images.to(cuda_device))
                    pred = torch.max(outp, dim=1)[1]
                    acc += torch.eq(pred, val_labels.to(cuda_device)).sum().item()

                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, args.epochs)
                
            acc /= val_num

            writer.add_scalar('val/acc', acc, epoch)

        print('[epoch %d] train_loss: %.5f  val_accuracy: %.5f' %
        (epoch + 1, total_loss / train_num, acc))
        
    print("Finished Training!")
    torch.save(model.state_dict(), args.save_path)

def test():
    model.eval()
    acc = 0.0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for test_data in test_bar:
            test_images, test_labels = test_data
            outp = model(test_images.to(cuda_device))
            pred = torch.max(outp, dim=1)[1]
            acc += torch.eq(pred, test_labels.to(cuda_device)).sum().item()

        acc /= test_num
        print("test acc = {:.5f}".format(acc))


if __name__ == '__main__':
    args = get_args()
    writer = SummaryWriter(log_dir=args.log_dir)
    cuda_device = args.cuda
    with torch.cuda.device(int(cuda_device[-1])):
        # 设置随机种子
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)

        if args.model == 'resnet34':
            model = resnet34(pretrained=args.pretrain).cuda()
        elif args.model == 'resnet18':
            model = resnet18(pretrained=args.pretrain).cuda()
        elif args.model == 'resnet50':
            model = resnet50(pretrained=args.pretrain).cuda()
        elif args.model == 'resnet101':
            model = resnet101(pretrained=args.pretrain).cuda()
        elif args.model == 'resnet152':
            model = resnet152(pretrained=args.pretrain).cuda()
        elif args.model == 'resnext50_32x4d':
            model = resnext50_32x4d(pretrained=args.pretrain).cuda()
        elif args.model == 'resnext101_32x8d':
            model = resnext101_32x8d(pretrained=args.pretrain).cuda()
        elif args.model == 'wide_resnet50_2':
            model = wide_resnet50_2(pretrained=args.pretrain).cuda()
        elif args.model == 'wide_resnet101_2':
            model = wide_resnet101_2(pretrained=args.pretrain).cuda()

        # 修改channel数（1000分类->10分类）
        in_channel = model.fc.in_features
        model.fc = nn.Linear(in_channel, 10)
        
        model.to(cuda_device)
        
        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.46252236, 0.45794976, 0.4298656], [0.24127056, 0.23532772, 0.24338122])]),
            "test": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.46252236, 0.45794976, 0.4298656], [0.24127056, 0.23532772, 0.24338122])])}

        data_root = os.path.abspath(os.path.join(
            os.getcwd(), ".."))  
        image_path = os.path.join(
            data_root, "imagenette2-320")  
        assert os.path.exists(
            image_path), "{} path does not exist.".format(image_path)

        if args.mode == 'adjust_parameter':
            # 建立损失函数
            loss_func = nn.CrossEntropyLoss()
            loss_func.to(cuda_device)
        
            # 建立优化器
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(params, lr=args.lr, eps=args.eps)

            
            whole_set = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                            transform=data_transform["train"])
            #划分validation set
            val_num = len(whole_set) * 0.2
            train_num = len(whole_set) - val_num
            train_set, validate_set = torch.utils.data.random_split(whole_set, [train_num, test_num])
            
            number_worker = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
            print('Using {} dataloader workers every process'.format(number_worker))

            train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=number_worker)
            val_loader = torch.utils.data.DataLoader(validate_set,
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=number_worker)
            train_num = len(train_set)
            val_num = len(validate_set)
            print("using {} images for training, {} images for validation.".format(train_num, val_num))
            #pdb.set_trace()
            train()

        if args.mode == 'train':
            # 建立损失函数
            loss_func = nn.CrossEntropyLoss()
            loss_func.to(cuda_device)
        
            # 建立优化器
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(params, lr=args.lr, eps=args.eps)
            
            train_set = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                            transform=data_transform["train"])
            
            number_worker = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
            print('Using {} dataloader workers every process'.format(number_worker))

            train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=number_worker)
            
            train_num = len(train_set)
            print("using {} images for training.".format(train_num))
            #pdb.set_trace()
            train()

        if args.mode == 'test':
            
            model_params_path = args.model_params_path
            assert os.path.exists(model_params_path), "file {} does not exist.".format(model_params_path)
            model.load_state_dict(torch.load(model_params_path, map_location=cuda_device))
        
            test_set = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform["test"])
            
            number_worker = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
            print('Using {} dataloader workers every process'.format(number_worker))

            test_loader = torch.utils.data.DataLoader(test_set,
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=number_worker)
            test_num = len(test_set)
            print("using {} images for testing.".format(test_num))

            test()
        
        
        

