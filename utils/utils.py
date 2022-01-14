import os
import numpy as np
from skimage import measure
from scipy import ndimage
import torch
import time
from utils import Bar, Logger, AverageMeter
import csv


def post_process_evaluate(x, target, name, args):

    JA_sum, AC_sum, DI_sum, SE_sum, SP_sum = [], [], [], [], []

    for i in range(x.shape[0]):
        x_tmp = x[i]
        target_tmp = target[i]
        x_tmp[x_tmp >= 0.5] = 1
        x_tmp[x_tmp < 0.5] = 0
        x_tmp = np.array(x_tmp, dtype='uint8')
        x_tmp = ndimage.binary_fill_holes(x_tmp).astype(int)

        box = []
        [lesion, num] = measure.label(x_tmp, return_num=True)
        if num == 0:
            JA_sum.append(0)
            AC_sum.append(0)
            DI_sum.append(0)
            SE_sum.append(0)
            SP_sum.append(0)
        else:
            region = measure.regionprops(lesion)
            for i in range(num):
                box.append(region[i].area)
            label_num = box.index(max(box)) + 1
            lesion[lesion != label_num] = 0
            lesion[lesion == label_num] = 1

            #  calculate TP,TN,FP,FN
            TP = float(np.sum(np.logical_and(lesion == 1, target_tmp == 1)))
            # True Negative (TN): we predict1 a label of 0 (negative), and the true label is 0.
            TN = float(np.sum(np.logical_and(lesion == 0, target_tmp == 0)))

            # False Positive (FP): we predict1 a label of 1 (positive), but the true label is 0.
            FP = float(np.sum(np.logical_and(lesion == 1, target_tmp == 0)))

            # False Negative (FN): we predict1 a label of 0 (negative), but the true label is 1.
            FN = float(np.sum(np.logical_and(lesion == 0, target_tmp == 1)))

            #  calculate JA, Dice, SE, SP
            JA = TP / ((TP + FN + FP + 1e-7))
            AC = (TP + TN) / ((TP + FP + TN + FN + 1e-7))
            DI = 2 * TP / ((2 * TP + FN + FP + 1e-7))
            SE = TP / (TP + FN+1e-7)
            SP = TN / ((TN + FP+1e-7))

            JA_sum.append(JA); AC_sum.append(AC); DI_sum.append(DI); SE_sum.append(SE); SP_sum.append(SP)

    if args.evaluate:
        f = open("/".join(args.resume.split('/')[:-1]) + '/result.csv', 'a', encoding='utf-8', newline='')
        csv_writer = csv.writer(f)
        #with open("/".join(args.resume.split('/')[:-1])+'/result.csv', 'a') as f:
        for item in range(0, len(DI_sum)):
            '''
            f.write("\n%s " % name[item].split("/")[-1])
            f.write("JA %.4f " % JA_sum[item])
            f.write("DI %.4f " % DI_sum[item])
            f.write("AC %.4f " % AC_sum[item])
            f.write("SE %.4f " % SE_sum[item])
            f.write("SP %.4f " % SP_sum[item])
            '''
            csv_writer.writerow(["baseline", str(JA_sum[item]), str(DI_sum[item]), str(AC_sum[item]), str(SE_sum[item]), str(SP_sum[item])])
        f.close()


    return sum(JA_sum), sum(AC_sum), sum(DI_sum), sum(SE_sum), sum(SP_sum)

def post_process_evaluate_pre(x, target):

    JA_sum, AC_sum, DI_sum, SE_sum, SP_sum = [], [], [], [], []


    x_tmp = x
    target_tmp = target


    x_tmp[x_tmp >= 0.5] = 1
    x_tmp[x_tmp < 0.5] = 0
    x_tmp = np.array(x_tmp, dtype='uint8')
    x_tmp = ndimage.binary_fill_holes(x_tmp).astype(int)

    # only reserve largest connected component.
    box = []
    [lesion, num] = measure.label(x_tmp, return_num=True)
    if num == 0:
        JA_sum.append(0)
        AC_sum.append(0)
        DI_sum.append(0)
        SE_sum.append(0)
        SP_sum.append(0)
    else:
        region = measure.regionprops(lesion)
        for i in range(num):
            box.append(region[i].area)
        label_num = box.index(max(box)) + 1
        lesion[lesion != label_num] = 0
        lesion[lesion == label_num] = 1

        #  calculate TP,TN,FP,FN
        TP = float(np.sum(np.logical_and(lesion == 1, target_tmp == 255)))
        # True Negative (TN): we predict1 a label of 0 (negative), and the true label is 0.
        TN = float(np.sum(np.logical_and(lesion == 0, target_tmp == 0)))

        # False Positive (FP): we predict1 a label of 1 (positive), but the true label is 0.
        FP = float(np.sum(np.logical_and(lesion == 1, target_tmp == 0)))

        # False Negative (FN): we predict1 a label of 0 (negative), but the true label is 1.
        FN = float(np.sum(np.logical_and(lesion == 0, target_tmp == 255)))

        #  calculate JA, Dice, SE, SP
        JA = TP / ((TP + FN + FP + 1e-7))
        AC = (TP + TN) / ((TP + FP + TN + FN + 1e-7))
        DI = 2 * TP / ((2 * TP + FN + FP + 1e-7))
        SE = TP / (TP + FN+1e-7)
        SP = TN / ((TN + FP+1e-7))

        JA_sum.append(JA); AC_sum.append(AC); DI_sum.append(DI); SE_sum.append(SE); SP_sum.append(SP)

    return sum(JA_sum), sum(AC_sum), sum(DI_sum), sum(SE_sum), sum(SP_sum)


def multi_validate(valloader, model, criterion, epoch, use_cuda, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = []
    label = []
    pred = []
    JA = 0
    DI = 0
    AC = 0
    SE = 0
    SP = 0
    i = 0


    # val_ground = np.vstack(val_ground)
    # print (val_ground.shape)
    # switch to evaluate mode
    model.eval()

    start = time.time()
    #bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets, name) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - start)

            if use_cuda:
                targets = targets.long()
                #targets[targets==255] = 1
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # init
            score_box = np.zeros((inputs.shape[0], 248, 248), dtype='float32')
            num_box = np.zeros((inputs.shape[0], 248, 248), dtype='uint8')

            # compute output
            outputs = model(inputs[:, :, 0:224, 0:224], dropout=False)
            Lx1 = criterion(outputs, targets[:, 0:224, 0:224].long())
            outputs = torch.softmax(outputs, dim=1)
            score_box[:, 0:224, 0:224] = outputs[:, 1, :, :].cpu().detach().numpy()
            num_box[:, 0:224, 0:224] = 1

            outputs = model(inputs[:, :, 24:248, 24:248], dropout=False)
            Lx2 = criterion(outputs, targets[:, 24:248, 24:248].long())
            outputs = torch.softmax(outputs, dim=1)
            score_box[:, 24:248, 24:248] += outputs[:, 1, :, :].cpu().detach().numpy()
            num_box[:, 24:248, 24:248] += 1

            outputs = model(inputs[:, :, 0:224, 24:248], dropout=False)
            Lx3 = criterion(outputs, targets[:, 0:224, 24:248].long())
            outputs = torch.softmax(outputs, dim=1)
            score_box[:, 0:224, 24:248] += outputs[:, 1, :, :].cpu().detach().numpy()
            num_box[:, 0:224, 24:248] += 1

            outputs = model(inputs[:, :, 24:248, 0:224], dropout=False)
            Lx4 = criterion(outputs, targets[:, 24:248, 0:224].long())
            outputs = torch.softmax(outputs, dim=1)
            score_box[:, 24:248, 0:224] += outputs[:, 1, :, :].cpu().detach().numpy()
            num_box[:, 24:248, 0:224] += 1

            score = score_box / (num_box + 1e-5)

            loss = (Lx1 + Lx2 + Lx3 + Lx4) / 4.0
            losses.append(loss.item())

            # measure accuracy and record loss
            x = score


            # y = val_ground[batch_idx]
            # x = cv2.resize(x[0], (y.shape[1], y.shape[0]),
            #                      interpolation=cv2.INTER_NEAREST)


            y = targets.cpu().detach().numpy()
            label.append(targets.cpu().detach().numpy().flatten())
            pred.append(x.flatten())
            results = post_process_evaluate(x, y, name, args)

            JA += results[0]
            AC += results[1]
            DI += results[2]
            SE += results[3]
            SP += results[4]

            i = i + inputs.shape[0]

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

    return [JA/i, AC/i,DI/i, SE/i, SP/i]


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)



def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    dice = (2. * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1. - dice
    return loss
