# README
task: White-box and Black-box attack
Dataset：CIFAR-10

Content: 
Write write-box and black-box attack codes by yourself. Then load a pretrained model and evaluate it under the above attack. You can follow the following steps:
	- Finish the code for FGSM and PGD attack with $L_\infty$ constraint.
	- Choose one black-box attack method and finish its code.
	- Load the attached pretrained model (wrn for CIFAR-10) and evaluate the performance under the above attacks. 

Hyper-parameters for CIFAR-10:
The limit on the perturbation size is ϵ=0.031 for both white-box and black-box attack.
The inner iteration for PGD is 10.
You should choose the step-size for PGD by yourself to reach the following success rate.