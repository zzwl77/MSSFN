from __future__ import print_function

import argparse
import os
import shutil
import time
import datetime
import random
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim import lr_scheduler
from scipy import stats

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import numpy as np
from AD_Dataloader import ADG_Dataloader, ADGH_Dataloader
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, LRschedule


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='PyTorch AD Training')

# 数据集
parser.add_argument('-d', '--dataset', default='D:\\1PHD\methods\\3method_trans\gazedata240828', type=str)
parser.add_argument('--datacsv', default='adnc_10f4', type=str, help='dataset .csv file')
parser.add_argument('--questionnaire', default='cls', type=str)
parser.add_argument('--n_fold', default=0, type=int, help='n-fold cross valodation')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='数据加载工作进程数 (默认: 4)')
parser.add_argument('--save_file', default='adghv_n60k12.pt', type=str)

# 优化选项
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='总共运行的训练周期数')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='手动指定的训练周期数 (用于重新开始)')
parser.add_argument('--train-batch', default=2, type=int, metavar='N',
                    help='训练批次大小')
parser.add_argument('--test-batch', default=2, type=int, metavar='N',
                    help='测试批次大小')
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                    metavar='LR', help='初始学习率')   
parser.add_argument('--T_max', default=30, type=int,
                    help='余弦退火调度器的最大迭代次数')
parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20],
                        help='在这些训练周期时降低学习率')
parser.add_argument('--gamma', type=float, default=0.5, help='在调度器中将学习率乘以的因子')
parser.add_argument('--step_size', type=int, default=5, help='LR调度器的步长')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='动量')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='权重衰减 (默认: 1e-4)')
parser.add_argument('--patience', default=20, type=int, 
                    help='LR调度器的耐心')

# 模型架构
parser.add_argument('--arch', '-a', metavar='ARCH', default='adg6',
                    choices=model_names,
                    help='模型架构: ' +
                        ' | '.join(model_names) +
                        ' (默认: resnet18)')

# 杂项选项
parser.add_argument('--manualSeed', type=int, help='手动随机种子')
parser.add_argument('-e', '--evaluate', dest='evaluate', type=bool, default=True,
                    help='仅在验证集上评估模型')
parser.add_argument('--test_ckp', default='checkpoint/model_best.pth.tar', type=str, metavar='PATH',
                    help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='最新检查点的路径 (默认: checkpoint/model_best.pth.tar)')
parser.add_argument('-l', '--logs', default='logs', type=str, metavar='PATH',
                    help='保存检查点的路径 (默认: logs)')
parser.add_argument('--best_acc', default=0.75, type=float)
#Device options
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
torch.cuda.set_device(int(args.gpu_id))
print('Use GPU: {} for training'.format(args.gpu_id))
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

def main():
    best_test_loss = float('inf')  # 设定初始的最佳测试损失为无穷大
    best_epoch = None
    best_acc = 0  # best test accuracy
    best_auc = 0
    patience_counter = 0
    ...
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    date = datetime.datetime.now().strftime("%y%m%d%H%M")
    log_dir = os.path.join(args.logs, args.datacsv, args.questionnaire, args.arch, str(args.n_fold), date)
    if not os.path.isdir(log_dir):
        mkdir_p(log_dir)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform = transforms.Compose([
            transforms.Resize((224, 224)),            
            # transforms.Grayscale(num_output_channels=1),  # 转换为灰度图，如果需要的话
            transforms.ToTensor()
            # transforms.Normalize((0.0044, 0.0102, 0.0289), (0.0467, 0.0646, 0.0993))
            ])
    ETdata = pd.read_csv(os.path.join(args.dataset, args.datacsv + '_u.csv'))
    full_dataset = ADGH_Dataloader(root = args.dataset, data_list=ETdata, label=args.questionnaire, transform=transform, save_file=args.save_file)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    # 按顺序将数据集分成 total_folds 份
    folds = np.array_split(indices, 5)
    
    # 取第 n_fold 作为测试集，其余作为训练集
    test_idx = folds[args.n_fold]
    print("fold_indexs: ", args.n_fold)
    train_idx = [idx for fold in folds if fold is not test_idx for idx in fold]
    
    train_subset = Subset(full_dataset, train_idx)
    test_subset = Subset(full_dataset, test_idx)
    
    trainloader = DataLoader(train_subset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = DataLoader(test_subset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    
    # Model
    print("==> creating model '{}'".format(args.arch))    
    model = models.__dict__[args.arch]().cuda()
    # model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_max, T_mult=2, eta_min=0, last_epoch=-1)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=-1)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma, last_epoch=-1)
    scheduler = LRschedule.Warmup_ExpDecayLR(optimizer=optimizer, warmup_epochs=0, total_epochs=args.epochs, warmup_lr=1e-4, peak_lr=3e-3, final_lr=1e-4)

    # Resume
    title = 'AD-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        log_dir = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger = Logger(os.path.join(log_dir, 'log.txt'), title=title, resume=True)
    else:
         # Print the names of the subfolders for 'AD' and 'NC' in the training and testing sets
        logger = Logger(os.path.join(log_dir, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Valid Loss',
                           'Train Acc.', 'Valid Acc.', 'Valid Sens.', 'Valid Spec.', 'Best Acc.', 'Best Epoch', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'])

    # if args.evaluate:
    #     checkpoint = torch.load(args.test_ckp)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     print('\nEvaluation only')
    #     _, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
    #     print('Test Acc:  %.2f' % (test_acc))
    #     return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        state['lr'] = scheduler.get_last_lr()[0]
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc, train_sensitivity, train_specificity = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        scheduler.step()

        test_loss, test_acc, test_sensitivity, test_specificity, test_precision, test_recall, test_f1, test_roc_auc = test(testloader, model, criterion, epoch, use_cuda)

        if args.patience:
            # early stopping
            if test_acc > best_acc:
                patience_counter = 0
            else:
                patience_counter += 1
            
            print('Patience counter: {}'.format(patience_counter))

        # append logger file
        is_best_acc = test_acc > best_acc
        is_best_auc = test_roc_auc > best_auc

        if is_best_acc:
            best_acc = test_acc
            best_epoch = epoch
        if is_best_auc:
            best_auc = test_roc_auc

        logger.append([epoch, state['lr'], train_loss, test_loss, train_acc, test_acc, test_sensitivity, test_specificity, best_acc, best_epoch, test_precision, test_recall, test_f1, test_roc_auc])
    
        # save model
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best_acc, is_best_auc, log_dir)
        
        if patience_counter >= args.patience:
                print("Early stopping!")
                break
        
    # fpr, tpr, _ = roc_curve(show_label, show_output)
    # roc_auc = auc(fpr, tpr)

    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC)')
    # plt.legend(loc="lower right")
    # plt.grid(True)
    # plt.savefig('ROC_Curve.png', dpi=300)
    # plt.close()
    # logger.plot(['Learning Rate'])
    # savefig(os.path.join(log_dir, 'lr.png'))

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    preds_all = []
    targets_all = []
    total_loss = 0.0
    start_time = time.time()

    bar = Bar('Train', max=len(trainloader))
    for batch_idx, (gazegraphs, heatmaps, taskmaps, age_edu, targets, image_path) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - start_time)
        if use_cuda:
            heatmaps = [heatmap.cuda() for heatmap in heatmaps]
            gazegraphs = [[sub.cuda() for sub in graph] for graph in gazegraphs]
            age_edu, targets = age_edu.cuda(), targets.cuda()
        outputs, contrastive_loss = model(heatmaps, gazegraphs, age_edu)
        preds = torch.sigmoid(outputs)

        loss1 = criterion(outputs.view(-1), targets.float())
        loss2 = contrastive_loss
        loss = loss1 + loss2
        total_loss += loss.item() 

        preds_all.append(preds.detach().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())

        losses.update(loss.item(), targets.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        # measure elapsed time
        batch_time.update(time.time() - start_time)

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()
    preds_binary = (np.concatenate(preds_all) > 0.5).astype(int)
    targets_binary = np.concatenate(targets_all).astype(int)
    acc = accuracy_score(targets_binary, preds_binary)
    sensitivity, specificity = cal_sens_spec(targets_binary, preds_binary)
    print('Total Time: {:.3f}s | ACC: {:.4f} | Sensitivity: {:.4f} | Specificity: {:.4f}'.format(time.time() - start_time, acc, sensitivity, specificity))
    return (losses.avg, acc, sensitivity, specificity)

def test(testloader, model, criterion, epoch, use_cuda, evaluate=False):
    model.eval()
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    preds_all = []
    targets_all = []
    total_loss = 0.0
    start_time = time.time()

    bar = Bar('Valid', max=len(testloader))
    with torch.no_grad():
        for batch_idx, (gazegraphs, heatmaps, taskmaps, age_edu, targets, image_path) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - start_time)

            if use_cuda:
                heatmaps = [heatmap.cuda() for heatmap in heatmaps]
                gazegraphs = [[sub.cuda() for sub in graph] for graph in gazegraphs]
                age_edu, targets = age_edu.cuda(), targets.cuda()
            outputs, contrastive_loss = model(heatmaps, gazegraphs, age_edu)
            preds = torch.sigmoid(outputs)

            loss1 = criterion(outputs.view(-1), targets.float())
            loss2 = contrastive_loss
            loss = loss1 + loss2            
            total_loss += loss.item() 

            preds_all.append(preds.detach().cpu().numpy())
            targets_all.append(targets.detach().cpu().numpy())  

            losses.update(loss.item(), targets.size(0))

            # measure elapsed time
            batch_time.update(time.time() - start_time)

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        )
            bar.next()
        bar.finish()
    
    # Convert predictions and targets to binary
    preds_binary = (np.concatenate(preds_all) > 0.5).astype(int)
    targets_binary = np.concatenate(targets_all).astype(int)
    # print(preds_binary, targets_binary)
    # Calculate accuracy, precision, recall, F1 score, and ROC-AUC
    acc = accuracy_score(targets_binary, preds_binary)
    precision = precision_score(targets_binary, preds_binary)
    recall = recall_score(targets_binary, preds_binary)
    f1 = f1_score(targets_binary, preds_binary)
    roc_auc = roc_auc_score(np.concatenate(targets_all), np.concatenate(preds_all))
    sensitivity, specificity = cal_sens_spec(targets_binary, preds_binary)

    # Calculate 95% confidence interval for sensitivity and specificity
    # sensitivity_conf_int = stats.t.interval(0.95, len(targets_binary)-1, loc=sensitivity, scale=stats.sem(targets_binary))
    # specificity_conf_int = stats.t.interval(0.95, len(targets_binary)-1, loc=specificity, scale=stats.sem(targets_binary))

    # print('Sensitivity 95% Confidence Interval: {:.4f} to {:.4f}'.format(sensitivity_conf_int[0], sensitivity_conf_int[1]))
    # print('Specificity 95% Confidence Interval: {:.4f} to {:.4f}'.format(specificity_conf_int[0], specificity_conf_int[1]))
    print('Total Time: {:.3f}s | ACC: {:.4f} | Sensitivity: {:.4f} | Specificity: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f} | ROC-AUC: {:.4f}'.format(time.time() - start_time, acc, sensitivity, specificity, precision, recall, f1, roc_auc))

    if evaluate:
        return preds_binary, targets_binary, image_path
    else:
        return losses.avg, acc, sensitivity, specificity, precision, recall, f1, roc_auc

def save_checkpoint(state, is_best_acc, is_best_auc, log_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(log_dir, filename)
    torch.save(state, filepath)
    if is_best_acc:
        shutil.copyfile(filepath, os.path.join(log_dir, 'model_best_acc.pth.tar'))
    if is_best_auc:
        shutil.copyfile(filepath, os.path.join(log_dir, 'model_best_auc.pth.tar'))

def cal_sens_spec(targets, preds):
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()

    # Calculate sensitivity (recall) and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

if __name__ == '__main__':
    main()