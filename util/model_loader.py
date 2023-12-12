
import os


import torch
from collections import OrderedDict

def get_model(args, num_classes, load_ckpt=True):
    if args.in_dataset == 'imagenet':
        if args.model_arch == 'resnet18':
            from models.resnet import resnet18
            model = resnet18(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'resnet50':
            from models.resnet import resnet50
            model = resnet50(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'mobilenet':
            from models.mobilenet import mobilenet_v2
            model = mobilenet_v2(num_classes=num_classes, pretrained=True)
    else:
        if args.model_arch == 'resnet50_Own':
            from models.resnet_Own import resnet50_Own
            model = resnet50_Own()
            model.load_state_dict(torch.load("checkpoints/CIFAR-100/resnet50/Sunday_26_March_2023_09h_13m_54s/resnet50-200-regular.pth"))
            print('hh')
        elif args.model_arch == 'resnet18':
            from models.resnet import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes, method=args.method)
        elif args.model_arch == 'resnet50':
            from models.resnet import resnet50_cifar
            model = resnet50_cifar(num_classes=num_classes, method=args.method)
        elif args.model_arch == 'wrn':
            from models.wrn import WideResNet
            model = WideResNet(depth=40, num_classes=100, widen_factor=2, dropRate=0.3, method=args.method)
        elif args.model_arch == 'densenet':
            from models.densenet import DenseNetBC100
            model = DenseNetBC100(num_c=10, method=args.method)
            print('hh')
        else:
            assert False, 'Not supported model arch: {}'.format(args.model_arch)

        # if load_ckpt:
        #     # checkpoint = torch.load("checkpoints/CIFAR-100/wrn/cifar100_wrn_pretrained_epoch_99.pt")
        #     # model.load_state_dict(checkpoint)
        #     model.load_state_dict(torch.load("./checkpoints/CIFAR-100/wrn/cifar100_wrn_pretrained_epoch_99.pt"))
            

    model.cuda()
    model.eval()
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    return model
