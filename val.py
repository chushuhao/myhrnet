import os
import torch
from torch import nn
from torch.utils import data
import numpy as np

from nets import seg_hrnet, seg_hrnet_cbam
from datasets import loveda
from utils.utils import FullModel, AverageMeter, get_confusion_matrix, kappa_score

num_class = 7
def create_model(modelname):
    model = eval(modelname + '.get_seg_model')(num_class)
    return model

def validate(valloader, model):
    model.eval()
    ave_loss = AverageMeter()
    confusion_matrix = np.zeros((num_class, num_class))
    with torch.no_grad():
        for idx, batch in enumerate(valloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()

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
    kappa = kappa_score(confusion_matrix)

    return ave_loss.average(), mean_IoU, IoU_array, acc, acc_array,kappa

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batchsize
    data_root = args.data_path
    val_data_list = os.path.join(data_root, 'val.lst')
    val_size = (1024, 1024)
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
        pin_memory=True, )
    model = create_model(modelname=args.modelname)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    model = FullModel(model, criterion)
    checkpoint = torch.load(args.path, map_location='cuda:0')
    model.model.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint.items() if k.startswith('model.')})
    print("=> train weight was loaded")
    valid_loss, mIoU, IoU_array, acc, acc_array, kappa = validate(valloader, model)
    msg = 'Loss: {:.3f}, MeanIoU: {: 4.4f}, acc: {: 4.4f}, kappa: {: 4.4f}'.format(
        valid_loss, mIoU, acc, kappa)
    print(msg)
    print('IoU_array = ' + ', '.join(IoU_array.astype(str)))
    print('acc_array = ' + ', '.join(acc_array.astype(str)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练网络
    parser.add_argument('--modelname', default='seg_hrnet_cbam', help='network')
    # 权重地址
    parser.add_argument('--path', default='./output/seg_hrnet_cbam/best_mIoU.pth',
                        help='path where to save the train weight')
    # 训练数据集的根目录
    parser.add_argument('--data-path', default='../../datasets/LoveDA', help='dataset')

    # 训练的batch size
    parser.add_argument('--batchsize', default=2, type=int, metavar='N',
                        help='batch size when training.')

    args = parser.parse_args()
    main(args)