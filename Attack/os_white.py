import os
os.system('python white_runner.py --cuda=cuda:2 --model=PreActResNet18 --epsilon=0.031 --noise --dataset=cifar --attack=mi_fgsm --stepsize=0.003 --loss=ce')
os.system('python white_runner.py --cuda=cuda:2 --model=PreActResNet18 --epsilon=0.031 --dataset=cifar --attack=mi_fgsm --stepsize=0.003 --loss=ce')
os.system('python white_runner.py --cuda=cuda:2 --model=PreActResNet18 --epsilon=0.031 --noise --dataset=cifar --attack=fgsm --loss=ce')
os.system('python white_runner.py --cuda=cuda:2 --model=PreActResNet18 --epsilon=0.031 --dataset=cifar --attack=fgsm --loss=ce')
os.system('python white_runner.py --cuda=cuda:2 --model=PreActResNet18 --epsilon=0.031 --noise --dataset=cifar --attack=pgd --stepsize=0.003 --loss=ce')
os.system('python white_runner.py --cuda=cuda:2 --model=PreActResNet18 --epsilon=0.031 --dataset=cifar --attack=pgd --stepsize=0.003 --loss=ce')

os.system('python white_runner.py --cuda=cuda:2 --model=PreActResNet18 --epsilon=0.031 --noise --dataset=cifar --attack=mi_fgsm --stepsize=0.003 --loss=nll')
os.system('python white_runner.py --cuda=cuda:2 --model=PreActResNet18 --epsilon=0.031 --dataset=cifar --attack=mi_fgsm --stepsize=0.003 --loss=nll')
os.system('python white_runner.py --cuda=cuda:2 --model=PreActResNet18 --epsilon=0.031 --noise --dataset=cifar --attack=fgsm --loss=nll')
os.system('python white_runner.py --cuda=cuda:2 --model=PreActResNet18 --epsilon=0.031 --dataset=cifar --attack=fgsm --loss=nll')
os.system('python white_runner.py --cuda=cuda:2 --model=PreActResNet18 --epsilon=0.031 --noise --dataset=cifar --attack=pgd --stepsize=0.003 --loss=nll')
os.system('python white_runner.py --cuda=cuda:2 --model=PreActResNet18 --epsilon=0.031 --dataset=cifar --attack=pgd --stepsize=0.003 --loss=nll')

os.system('python white_runner.py --cuda=cuda:2 --model=small_cnn --epsilon=0.3 --noise --dataset=MNIST --attack=mi_fgsm --stepsize=0.1 --loss=ce')
os.system('python white_runner.py --cuda=cuda:2 --model=small_cnn --epsilon=0.3 --dataset=MNIST --attack=mi_fgsm --stepsize=0.1 --loss=ce')
os.system('python white_runner.py --cuda=cuda:2 --model=small_cnn --epsilon=0.3 --noise --dataset=MNIST --attack=fgsm --loss=ce')
os.system('python white_runner.py --cuda=cuda:2 --model=small_cnn --epsilon=0.3 --dataset=MNIST --attack=fgsm --loss=ce')
os.system('python white_runner.py --cuda=cuda:2 --model=small_cnn --epsilon=0.3 --noise --dataset=MNIST --attack=pgd --stepsize=0.1 --loss=ce')
os.system('python white_runner.py --cuda=cuda:2 --model=small_cnn --epsilon=0.3 --dataset=MNIST --attack=pgd --stepsize=0.1 --loss=ce')

os.system('python white_runner.py --cuda=cuda:2 --model=small_cnn --epsilon=0.3 --noise --dataset=MNIST --attack=mi_fgsm --stepsize=0.1 --loss=nll')
os.system('python white_runner.py --cuda=cuda:2 --model=small_cnn --epsilon=0.3 --dataset=MNIST --attack=mi_fgsm --stepsize=0.1 --loss=nll')
os.system('python white_runner.py --cuda=cuda:2 --model=small_cnn --epsilon=0.3 --noise --dataset=MNIST --attack=fgsm --loss=nll')
os.system('python white_runner.py --cuda=cuda:2 --model=small_cnn --epsilon=0.3 --dataset=MNIST --attack=fgsm --loss=nll')
os.system('python white_runner.py --cuda=cuda:2 --model=small_cnn --epsilon=0.3 --noise --dataset=MNIST --attack=pgd --stepsize=0.1 --loss=nll')
os.system('python white_runner.py --cuda=cuda:2 --model=small_cnn --epsilon=0.3 --dataset=MNIST --attack=pgd --stepsize=0.1 --loss=nll')