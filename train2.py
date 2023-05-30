import json
import logging
import os
import datetime
import time
import timeit

import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import numpy as np

from nets import seg_hrnet, seg_hrnet_cbam
from datasets import loveda
from utils.utils import FullModel, AverageMeter, adjust_learning_rate, get_confusion_matrix, create_logger

num_class = 7    #分类的数量




def create_model(modelname, pretrain_path, load_pretrain_weights=True):
    model = eval(modelname + '.get_seg_model')(num_class)
    if load_pretrain_weights:
        model.init_weights(pretrain_path)

    return model

def train(epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model,device):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        losses, _ = model(images, labels)
        loss = losses.mean()
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % 100 == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())
            logging.info(msg)

def validate(valloader, model,device):
    model.eval()
    ave_loss = AverageMeter()
    confusion_matrix = np.zeros((num_class, num_class))
    with torch.no_grad():
        for idx, batch in enumerate(valloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)

            losses, pred = model(image, label)
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                num_class,
                ignore=255
            )

            loss = losses.mean()
            ave_loss.update(loss.item())

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    acc_array = tp/pos
    acc = tp.sum() / confusion_matrix.sum()

    return ave_loss.average(), mean_IoU, IoU_array, acc


def main(args):
    logger = create_logger(dataset='loveda', model=args.modelname, phase='train')
    logging.info(args)
    wd = args.wd
    batch_size = args.batchsize
    lr = args.lr
    final_output_file = os.path.join(args.weight_output_file, args.modelname, 'last_weight.pth')
    best_output_mIoU = os.path.join(args.weight_output_file, args.modelname, 'best_mIoU.pth')
    best_output_acc = os.path.join(args.weight_output_file, args.modelname, 'best_acc.pth')
    os.makedirs(os.path.dirname(final_output_file), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using {} device training.".format(device.type))
    # 加载数据
    data_root = args.data_path
    val_data_list = os.path.join(data_root, 'val.lst')
    train_data_lit = os.path.join(data_root, 'train.lst')
    crop_size = (1024, 1024)
    train_dataset = loveda(
        root=data_root,
        list_path=train_data_lit,
        num_classes=num_class,
        multi_scale=True,
        flip=True,
        ignore_label=255,
        base_size=1024,
        crop_size=crop_size,
        downsample_rate=1,
        scale_factor=11)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,)

    val_size = (1024,1024)
    val_dataset = loveda(
        root=data_root,
        list_path=val_data_list,
        num_classes=num_class,
        multi_scale=False,
        flip=False,
        ignore_label=255,
        base_size=1024,
        crop_size=val_size,
        downsample_rate=1)

    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,)

    # 创建模型
    model = create_model(modelname=args.modelname, pretrain_path=args.pretrain_path)

    # criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    model.to(device)

    model = FullModel(model, criterion)

    # optimizer
    params_dict = dict(model.named_parameters())
    params = [{'params': list(params_dict.values()), 'lr': lr}]
    optimizer = torch.optim.SGD(params,
                                lr=lr,
                                momentum=0.9,
                                weight_decay=wd,
                                nesterov=False,
                                )


    epoch_iters = int(train_dataset.__len__() / batch_size)
    best_mIoU = 0
    best_acc = 0
    last_epoch = 0
    if args.resume :
        model_state_file = final_output_file
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:1': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']

            model.model.load_state_dict(
                {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = args.endepoch #总训练轮数
    num_iters = end_epoch * epoch_iters

    for epoch in range(last_epoch, end_epoch):
        train(epoch, end_epoch,
                epoch_iters, lr, num_iters,
                trainloader, optimizer, model, device)

        valid_loss, mean_IoU, IoU_array, acc = validate(valloader, model, device)
        msg = 'Loss: {:.3f}, MeanIoU: {: 4.4f}, Best_mIoU: {: 4.4f}, acc: {: 4.4f}'.format(
            valid_loss, mean_IoU, best_mIoU,acc)
        logger.info(msg)

        logging.info('=> saving checkpoint to {}'.format(final_output_file))
        torch.save({
            'epoch': epoch + 1,
            'best_mIoU': best_mIoU,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, final_output_file)
        if mean_IoU > best_mIoU:
            logging.info('=> saving best mIoU checkpoint to {}'.format(best_output_mIoU))
            best_mIoU = mean_IoU
            torch.save(model.state_dict(), best_output_mIoU)
        if acc > best_acc:
            logging.info('=> saving best acc checkpoint to {}'.format(best_output_acc))
            best_acc = acc
            torch.save(model.state_dict(), best_output_acc)


    torch.save(model.state_dict(),final_output_file)

    end = timeit.default_timer()
    logger.info('Hours: %d' % int((end - start) / 3600))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:1', help='device')
    # 训练网络
    parser.add_argument('--modelname', default='seg_hrnet_cbam', help='network')
    # 预训练权重地址
    parser.add_argument('--pretrain_path', default='./hrnetv2_w48_imagenet_pretrained.pth',
                        help='path where to save the last-train weight')
    # 训练数据集的根目录
    parser.add_argument('--data-path', default='../../datasets/LoveDA', help='dataset')
    # 是否接着上次训练
    parser.add_argument('--resume', default=False, type=bool, help='resume or not')
    # 权重输出文件夹
    parser.add_argument('--weight_output_file', default='./output',
                        help='path where to save the last-train weight')
    # 训练的总epoch数
    parser.add_argument('--endepoch', default=50, type=int, metavar='N',
                        help='number of end epochs to run')
    # 学习率
    parser.add_argument('--lr', default=4e-3, type=float,
                        help='initial learning rate')
    # weight_decay参数
    parser.add_argument('--wd', '--weight_decay', default=0.0005, type=float,
                        metavar='W', help='weight decay',
                        dest='wd')
    # 训练的batch size
    parser.add_argument('--batchsize', default=2, type=int, metavar='N',
                        help='batch size when training.')

    args = parser.parse_args()
    main(args)