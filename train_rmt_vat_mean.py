from __future__ import print_function
import argparse
import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import utils.semantic_seg as transform
import models.network as models
from mean_teacher import losses, ramps
from utils import mkdir_p
from tensorboardX import SummaryWriter
from utils.utils import multi_validate, update_ema_variables, dice_loss
from utils.vat import VATLoss
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=1024, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                    help='train batchsize')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--n-labeled', type=int, default=50,
                    help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=10,
                    help='Number of labeled data')
parser.add_argument('--data', default='',
                    help='input data path')
parser.add_argument('--out', default='output/skin/skin50_tcsm/',
                    help='Directory to output the result')
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--num-class', default=2, type=int)
parser.add_argument('--evaluate', action="store_true")
parser.add_argument('--baseline', action="store_true")
parser.add_argument('--covid_ct', action="store_true")

# lr
parser.add_argument("--lr_mode", default="cosine", type=str)
parser.add_argument("--lr", default=0.03, type=float)
parser.add_argument("--warmup_epochs", default=0, type=int)
parser.add_argument("--warmup_lr", default=0.0, type=float)
parser.add_argument("--targetlr", default=0.0, type=float)

#
parser.add_argument('--consistency_type', type=str, default="mse")
parser.add_argument('--consistency', type=float,  default=10.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=400.0, help='consistency_rampup')

#
parser.add_argument('--initial-lr', default=0.0, type=float,
                    metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                    help='length of learning rate rampup in the beginning')
parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',
                    help='length of learning rate cosine rampdown (>= length of training)')
parser.add_argument('--momentum', default=0.99, type=float, metavar='M',
                    help='momentum')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)            # 为CPU设置随机种子
torch.cuda.manual_seed(args.manualSeed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(args.manualSeed) # 为所有GPU设置随机种子
os.environ['PYTHONHASHSEED'] = str(args.manualSeed)

best_ja = 0  # best predict1 accuracy
NUM_CLASS = args.num_class

from shutil import copyfile


def main():

    global best_ja

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    if args.covid_ct:
        mean = [0.245669, 0.245669, 0.245669]
        std = [0.075089, 0.075089, 0.075089]
    else:
        mean = [0.707647, 0.591440, 0.546651]
        std = [0.024013, 0.026675, 0.031657]
    # Data augmentation
    print(f'==> Preparing skinlesion dataset')
    transform_train = transform.Compose([
        transform.RandomRotationScale(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    transform_val = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    if args.covid_ct:
        import dataset.covid_ct as dataset
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_skinlesion_dataset("./data/covid_ct/",
                                                                                         num_labels=args.n_labeled,
                                                                                         transform_train=transform_train,
                                                                                         transform_val=transform_val,
                                                                                         transform_forsemi=None)
    else:
	    import dataset.skinlesion as dataset
		train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_skinlesion_dataset("./data/skinlesion/",
																								   num_labels=args.n_labeled,
																								   transform_train=transform_train,
																								   transform_val=transform_val,
																								   transform_forsemi=None)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True,
                                          num_workers=2, drop_last=True)
    if args.baseline:
        unlabeled_trainloader = None
    else:

        unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True,
                                                num_workers=2, drop_last=True)

    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    #test_loader = data.DataLoader(test_set, batch_size=args.batch-size, shuffle=False, num_workers=2)



    # Model
    print("==> creating model")

    def create_model(ema=False):
        model = models.DenseUnet_2d()
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)


    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    if args.covid_ct:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.36, 9.12]).cuda())
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.55, 3.89]).cuda())

    vat_loss = VATLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0

    # Resume
    if args.resume:
        print('==> Resuming from checkpoint..' + args.resume)
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        best_ja = checkpoint['best_ja']
        print("epoch ", checkpoint['epoch'])
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.evaluate:
        
        val_result = multi_validate(val_loader, ema_model, criterion, 0, use_cuda, args)
        print("Val ema_model : JA, AC, DI, SE, SP \n")
        print(", ".join("%.4f" % f for f in val_result))
      
		val_result = multi_validate(val_loader, model, criterion, 0, use_cuda, args)
        print("Val model: JA, AC, DI, SE, SP \n")
        print(", ".join("%.4f" % f for f in val_result))

        return

    writer = SummaryWriter("runs/" + str(args.out.split("/")[-1]))
    writer.add_text('Text', str(args))

    for epoch in range(start_epoch, args.epochs):
        val_result = multi_validate(val_loader, model, criterion, epoch, use_cuda, args)
        val_ema_result = multi_validate(val_loader, ema_model, criterion, epoch, use_cuda, args)

        step = args.val_iteration * (epoch)

        writer.add_scalar('Model/JA', val_result[0], step)
        writer.add_scalar('Model/AC', val_result[1], step)
        writer.add_scalar('Model/DI', val_result[2], step)
        writer.add_scalar('Model/SE', val_result[3], step)
        writer.add_scalar('Model/SP', val_result[4], step)


        writer.add_scalar('Ema_model/JA', val_ema_result[0], step)
        writer.add_scalar('Ema_model/AC', val_ema_result[1], step)
        writer.add_scalar('Ema_model/DI', val_ema_result[2], step)
        writer.add_scalar('Ema_model/SE', val_ema_result[3], step)
        writer.add_scalar('Ema_model/SP', val_ema_result[4], step)
        # scheduler.step()

        # save model
        big_result = max(val_result[0], val_ema_result[0])
        is_best = big_result > best_ja
        best_ja = max(big_result, best_ja)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'ja': val_result[0],
            'best_ja': best_ja,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        # train
        train_meanteacher(labeled_trainloader, unlabeled_trainloader, model, ema_model, optimizer,
                          criterion, epoch, writer, vat_loss)



    writer.close()

    print('Best JA:')
    print(best_ja)

def train_meanteacher(labeled_trainloader, unlabeled_trainloader, model, ema_model, optimizer,
                    criterion, epoch, writer, vat_loss):
    global global_step
    print("train meanteacher!!!")
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    # switch to train mode
    model.train()
    ema_model.train()

    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x, name_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, name_x = labeled_train_iter.next()

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)

        if not args.baseline:

            try:
                inputs_u, _ = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, _ = unlabeled_train_iter.next()

            if use_cuda:
                # targets_x[targets_x == 255] = 1
                inputs_u = inputs_u.cuda()



        # iter_num
        iter_num = batch_idx + epoch * args.val_iteration
        
		#calculate lds 
		lds = vat_loss(model, ema_model, inputs_u)

        # labeled data
        logits_x = model(inputs_x)
        Lx = criterion(logits_x, targets_x.long())
        '''
        outputs_soft = F.softmax(logits_x, dim=1)
        Lx_dice = dice_loss(outputs_soft[:, 1, :, :], targets_x.long())
        Lx = 0.4 * Lx_ce + 0.6 * Lx_dice
        '''
        # unlabeled data
        if not args.baseline:
            consistency_weight = get_current_consistency_weight(epoch)
            Lu = consistency_weight * lds

            loss = Lu + Lx
        else:
            loss = Lx

        print("loss=", loss.detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, args.ema_decay, iter_num)
        '''
        lr_ = args.lr * (1 - iter_num / (args.val_iteration * args.epochs)) ** 0.9

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        writer.add_scalar('lr', lr_, (epoch) * args.val_iteration)
        '''
        writer.add_scalar('losses/train_loss', loss, iter_num)
        writer.add_scalar('losses/train_loss_supervised', Lx, iter_num)
        if not args.baseline:
            writer.add_scalar('losses/train_loss_un', Lu, iter_num)
            writer.add_scalar('losses/consistency_weight', consistency_weight, iter_num)

    print("-" * 50)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def save_checkpoint(state, is_best, checkpoint=args.out, filename='rmt_vat_checkpoint_50.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'rmt_vat_model_best_50.pth.tar'))


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


if __name__ == '__main__':
   main()