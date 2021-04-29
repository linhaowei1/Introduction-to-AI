try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-1,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=350,
                        help='upper epoch limit')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='cuda used for training')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--mode', type=str, default='train',
                        help='choose the mode for runner. mode availiable = train/test')
    parser.add_argument('--pretrain', dest='pretrain', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--save_path', type=str, default='./resNet50(from_scratch).pth',
                        help='save the model here')
    parser.add_argument('--model_params_path', type=str, default='./resnet50-pre.pth',
                        help='use pretrained model from here')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='term added to the denominator to improve numerical stability')
    parser.add_argument('--log_dir', type=str, default='from_scratch_resnet50',
                        help='save logs')
    parser.add_argument('--model', type=str, default='PreActResNet18',
                        help='use model')
    parser.add_argument('--epsilon', type=float, default=0.031,
                        help='the attack epsilon')
    parser.add_argument('--noise', dest='noise', action='store_true',
                        help='use random noise to attack')
    parser.add_argument('--loss', type=str, default='ce',
                        help='use loss')
    parser.add_argument('--dataset', type=str, default='cifar',
                        help='which dataset to use')
    parser.add_argument('--attack', type=str, default='fgsm',
                        help='which attack method to use')
    # MNIST 用0.25， CIFAR用0.003
    parser.add_argument('--stepsize', type=float, default=0.003,
                        help='stepsize')
    parser.add_argument('--step_number', type=int, default=10,
                        help='step number')
    parser.add_argument('--weight_decay', '-w', default=5e-4, type=float, help='weight_decay')

    return parser.parse_args()