"""
The code in this file is built on thr top of OPTQ, please visit:
https://github.com/IST-DASLab/gptq
for their origin contribution

SPDX-License-Identifier: Apache-2.0

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (2024) Bytedance Ltd. and/or its affiliates.
"""
import time

import torch
import torch.nn as nn

from decoupleQ.quant import find_layers
import os
from datetime import timedelta
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import random
import warnings
import torch.backends.cudnn as cudnn
from enum import Enum
from decoupleQ.quant import decoupleQ, replace_forward, recover_forward
from decoupleQ.moq_quant import Quantizer

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def get_resnet(model):
    # model = models.__dict__[args.model](weights=ResNet18_Weights.DEFAULT)
    model = models.__dict__[args.model](pretrained=True)

    return model


quant_res18 = ['conv1', 'layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1',
               'layer1.1.conv2', 'layer2.0.conv1', 'layer2.0.conv2', 'layer2.0.downsample.0', 'layer2.1.conv1',
               'layer2.1.conv2', 'layer3.0.conv1', 'layer3.0.conv2', 'layer3.0.downsample.0', 'layer3.1.conv1',
               'layer3.1.conv2', 'layer4.0.conv1', 'layer4.0.conv2', 'layer4.0.downsample.0', 'layer4.1.conv1',
               'layer4.1.conv2', 'fc']

quant_res50 = ['conv1', 'layer1.0.conv1', 'layer1.0.conv2', 'layer1.0.conv3',
               'layer1.0.downsample.0', 'layer1.1.conv1', 'layer1.1.conv2', 'layer1.1.conv3',
               'layer1.2.conv1', 'layer1.2.conv2', 'layer1.2.conv3', 'layer2.0.conv1', 'layer2.0.conv2',
               'layer2.0.conv3', 'layer2.0.downsample.0', 'layer2.1.conv1', 'layer2.1.conv2', 'layer2.1.conv3',
               'layer2.2.conv1', 'layer2.2.conv2', 'layer2.2.conv3', 'layer2.3.conv1', 'layer2.3.conv2',
               'layer2.3.conv3', 'layer3.0.conv1', 'layer3.0.conv2', 'layer3.0.conv3', 'layer3.0.downsample.0',
               'layer3.1.conv1', 'layer3.1.conv2', 'layer3.1.conv3', 'layer3.2.conv1', 'layer3.2.conv2',
               'layer3.2.conv3', 'layer3.3.conv1', 'layer3.3.conv2', 'layer3.3.conv3', 'layer3.4.conv1',
               'layer3.4.conv2', 'layer3.4.conv3', 'layer3.5.conv1', 'layer3.5.conv2', 'layer3.5.conv3',
               'layer4.0.conv1', 'layer4.0.conv2', 'layer4.0.conv3', 'layer4.0.downsample.0',
               'layer4.1.conv1', 'layer4.1.conv2', 'layer4.1.conv3', 'layer4.2.conv1', 'layer4.2.conv2',
               'layer4.2.conv3', 'fc']


@torch.no_grad()
def resnet_sequential(args, model, dataloader, dev):
    subset = find_layers(model)
    if args.model == "resnet18":
        quant_layers = quant_res18
    elif args.model == "resnet50":
        quant_layers = quant_res50
    else:
        raise NotImplementedError("not support yet")
    quantizers = {}
    model = model.train()
    for name in quant_layers:
        print(name)
        print('Quantizing ...')
        t1 = time.time()
        moq = decoupleQ(subset[name])
        moq.quantizer = Quantizer()
        moq.quantizer.configure(args.wbits, perchannel=True, sym=args.sym)
        subset[name].mask = [None]

        def add_batch():
            def tmp(module, inp, out):
                moq.add_batch(inp[0].data, out.data, module.mask[0])
                raise ValueError("exit early")

            return tmp

        handles = [subset[name].register_forward_hook(add_batch())]
        for i, (image, labels) in enumerate(dataloader):
            image = image.to(dev)
            try:
                model(image)
            except ValueError:
                pass
        handles[0].remove()
        del image, labels

        scale_out, zero_out, w_int, loss = moq.startquant(
            dev=dev,
            groupsize=args.group_size,
            symmetric=args.sym,
            max_iter_num=args.max_iter_num,
            inner_iters_for_round=args.inner_iters_for_round,
            iters_before_round=args.iters_before_round,
            lr=args.lr,
            actorder=args.act_order,
            round_fn=args.round_fn,
        )
        t2 = time.time()
        print(f"{name}, time cost {t2 - t1}, loss is {loss.mean().item()}")
        print()
        scale_list = [k.cpu() for k in [scale_out, zero_out]]
        quantizers[f"{name}.weight"] = {
            "scales": scale_list, "weights": w_int.cpu(), "loss": loss.cpu()}
        moq.free()
        moq.quantizer.free()
        del moq, scale_out, zero_out, w_int
    return quantizers


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


def validate(val_loader, model, criterion, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                images = images.cuda(dev, non_blocking=True)
                target = target.cuda(dev, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    progress.display_summary()

    return top1.avg


def recal_bn(model, dataloader, dev):
    momentum = {}
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            momentum[name] = module.momentum
            module.momentum = None
            module.reset_running_stats()

    model.train()
    with torch.no_grad():
        for i, (image, label) in enumerate(dataloader):
            image = image.to(dev)
            model(image)

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            module.momentum = momentum[name]


def sft(args, model, dev, quantizers, train_loader):
    model = model.to(dev)
    full = find_layers(model)
    params = []
    for key in full:
        quantizer = quantizers[f"{key}.weight"]
        weight = quantizer["weights"]
        scale_list = quantizer["scales"]
        dtype = torch.float32
        factory_kwargs = {'device': dev, 'dtype': dtype}
        full[key].weight.data = weight.to(**factory_kwargs)
        full[key].weight.requires_grad_(False)
        scale = torch.nn.Parameter(scale_list[0].clone().to(**factory_kwargs), requires_grad=True)
        requires_grad = True if args.asym else False
        zero = torch.nn.Parameter(scale_list[1].clone().to(**factory_kwargs), requires_grad=requires_grad)
        full[key].register_parameter("scale", scale)
        full[key].register_parameter("zero", zero)
        full[key].group_size = args.group_size
        params.append(scale)
        if args.asym:
            params.append(zero)

    if args.train_LN:
        for k, m in model.named_modules():
            if isinstance(m, (torch.nn.LayerNorm, torch.nn.BatchNorm2d)):
                if hasattr(m, "weight"):
                    m.weight.requires_grad_(True)
                    params.append(m.weight)
                    print(f"add {k} layer norm weight to train")
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad_(True)
                    params.append(m.bias)
                    print(f"add {k} layer norm bias to train")

    origin_forward = replace_forward(model)
    lr = args.blockwise_minimize_lr
    # optimizer = torch.optim.Adam(params, lr, eps=2.e-5, betas=(0.9, 0.99), weight_decay=args.blockwise_minimize_wd)
    optimizer = torch.optim.SGD(params, lr, momentum=0.9, weight_decay=args.blockwise_minimize_wd)
    print("--", optimizer.param_groups[0]["lr"])
    for epoch in range(args.blockwise_minimize_epoch):
        train(train_loader, model, criterion, optimizer, epoch, dev, args)
        acc1 = validate(val_loader, model, criterion, args)
        print(f"epoch {epoch}, acc is {acc1}")
        optimizer.param_groups[0]["lr"] *= 0.5

    recover_forward(model, origin_forward)
    for key in full:
        scale, zero = full[key].scale, full[key].zero
        quantizers[f"{key}.weight"]["scales"] = [scale.cpu(), zero.cpu()]
        shape = [full[key].weight.shape[0]] + (full[key].weight.ndim - 1) * [1]
        scale = torch.reshape(scale, shape=shape)
        zero = torch.reshape(zero, shape=shape)
        full[key].weight.data = full[key].weight.data * scale + zero
        del full[key].scale, full[key].zero

    return quantizers


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i + 1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        '--dataset', type=str,
        help='Where to extract calibration data from.'
    )
    parser.add_argument('--multiprocessing-distributed', action='store_true', default=True,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument(
        '--seed',
        type=int, default=1, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=0.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--group-size', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--faster-kernel', action='store_true',
        help='Whether to use the new faster kernel for benchmarking.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    parser.add_argument(
        '--loss-thr', type=float, default=0.02,
        help='The loss threshold to exit loop'
    )
    parser.add_argument(
        '--max-iter-num', type=int, default=3,
        help='The max iter num for the whole loop for quantization'
    )
    parser.add_argument(
        '--inner-iters-for-round', type=int, default=50,
        help='the number of iters for PGD when use first level approximation'
    )
    parser.add_argument(
        '--iters-before-round', type=int, default=0,
        help='the number of iters before entering PGD when use first level approximation'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='the learning rate for PGD'
    )
    parser.add_argument(
        '--round-fn', type=str, choices=["gptq", "train"], default="train",
        help='the quant method'
    )
    parser.add_argument(
        '--blockwise-minimize-lr', type=float, default=-1.0,
        help='the learning rate for block minimization'
    )
    parser.add_argument(
        '--blockwise-minimize-wd', type=float, default=1.0e-6,
        help='the weight decaying rate for block minimization'
    )
    parser.add_argument(
        '--blockwise-minimize-epoch', type=int, default=3,
        help='the number of epoch for training the float point part'
    )
    parser.add_argument(
        '--train-LN', action='store_true',
        help='Whether to train the parameters in norm'
    )
    args = parser.parse_args()
    args.asym = not args.sym
    args.qbits = args.wbits
    print(args)

    rank = 0
    dev = f"cuda:{rank}"
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    model = get_resnet(args.model).to(dev)

    traindir = os.path.join(args.dataset, 'train')
    valdir = os.path.join(args.dataset, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    batch_size = 1024
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, sampler=None)

    criterion = nn.CrossEntropyLoss().to(dev)
    # acc1 = validate(val_loader, model, criterion, args)
    # print(acc1)

    dataloader = []
    for i, (images, target) in enumerate(train_loader):
        dataloader.append((images, target))
        if i == args.nsamples - 1:
            break

    images = torch.cat([t[0] for t in dataloader])
    labels = torch.cat([t[1] for t in dataloader])
    res = [torch.sum(labels == i) / labels.shape[0] for i in range(1000)]
    del images, labels
    torch.cuda.empty_cache()
    print(min(res), max(res))
    t1 = time.time()
    quantizers = resnet_sequential(args, model, dataloader, dev)
    t2 = time.time()
    print(f"The duration is {(t2 - t1) / 3600}")

    acc1 = validate(val_loader, model, criterion, args)
    print("The acc after quantization is ", acc1)

    recal_bn(model, dataloader, dev)

    acc1 = validate(val_loader, model, criterion, args)
    print("The acc after recalBN is ", acc1)
    # print("The time for quant is", t2 - t1)
    if args.blockwise_minimize_lr > 0:
        # del dataloader
        torch.cuda.empty_cache()
        if args.model == "resnet50":
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size // 2, shuffle=True, num_workers=8, pin_memory=True)
        sft(args, model, dev, quantizers, train_loader)
        recal_bn(model, dataloader, dev)
        acc1 = validate(val_loader, model, criterion, args)
        print("The acc after recal bn is ", acc1)
