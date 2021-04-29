import os
#os.system('python black_runner.py --cuda=cuda:2 --model=small_cnn --dataset=MNIST --loss=ce --epochs=500 --lr=0.1 --pretrain')
os.system('python black_runner.py --cuda=cuda:2 --model=PreActResNet18 --dataset=cifar  --loss=ce --epoch=350 --lr=0.1 --pretrain')