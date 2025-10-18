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
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
from AD_Dataloader import ADG_Dataloader, ADGH_Dataloader
import models
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from utils.visualize import boxplot, boxplot2, plot_roc, violinplot2, plot_roc2

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='PyTorch AD Training')

# 数据集
parser.add_argument('-d', '--dataset', default='D:\\1PHD\\methods\\3method_trans\\gazedata240828', type=str)
parser.add_argument('--datacsv', default='adnc_10f4', type=str, help='dataset .csv file')
parser.add_argument('--questionnaire', default='cls', type=str)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='数据加载工作进程数 (默认: 4)')
# 优化选项
parser.add_argument('--test-batch', default=10, type=int, metavar='N',
                    help='测试批次大小')

# 杂项选项
parser.add_argument('--manualSeed', type=int, help='手动随机种子')
parser.add_argument('-e', '--evaluate', dest='evaluate', type=bool, default=True,
                    help='仅在验证集上评估模型')

parser.add_argument('--save_file', default='adghv.pt', type=str)
parser.add_argument('--aucname', default='SEA&ATT', type=str, help='auc name(Saccade task, Visual search task, Visual attention task, Multitask)')
# 模型架构
parser.add_argument('--arch', '-a', metavar='ARCH', default='adg6ns',
                    choices=model_names,
                    help='模型架构: ' +
                        ' | '.join(model_names) +
                        ' (默认: resnet18)')
parser.add_argument('--test_ckp', default=["D:\\1PHD\methods\\3method_trans\ADR240801\logs\\adnc_10f4\cls\\adg6\\0\\2411261045\model_best_acc.pth.tar",
                                           "D:\\1PHD\methods\\3method_trans\ADR240801\logs\\adnc_10f4\cls\\adg6\\1\\2411262055\model_best_acc.pth.tar",
                                           "D:\\1PHD\methods\\3method_trans\ADR240801\logs\\adnc_10f4\cls\\adg6\\2\\2411260756\model_best_acc.pth.tar",
                                           "D:\\1PHD\methods\\3method_trans\ADR240801\logs\\adnc_10f4\cls\\adg6\\3\\2411260951\model_best_acc.pth.tar",
                                           "D:\\1PHD\methods\\3method_trans\ADR240801\logs\\adnc_10f4\cls\\adg6\\4\\2411262223\model_best_acc.pth.tar"], type=str, nargs='+', metavar='PATH',
                    help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
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
# use_cuda = False

# Random seed
# if args.manualSeed is None:
#     args.manualSeed = random.randint(1, 10000)
# random.seed(args.manualSeed)
# torch.manual_seed(args.manualSeed)

# cudnn.benchmark = True

# if use_cuda:
#     torch.cuda.manual_seed_all(args.manualSeed)

def main():
    ...

    # Data
    print('==> Preparing dataset %s' % args.dataset)

    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
            # transforms.Normalize((0.0044, 0.0102, 0.0289), (0.0467, 0.0646, 0.0993))
            ])
    ETdata = pd.read_csv(os.path.join(args.dataset, args.datacsv + '_u.csv'))
    full_dataset = ADGH_Dataloader(root = args.dataset, data_list=ETdata, label=args.questionnaire, transform=transform, save_file=args.save_file)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    # 按顺序将数据集分成 total_folds 份
    folds = np.array_split(indices, 5)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if use_cuda:
        model = models.__dict__[args.arch]().cuda()
    else:
        model = models.__dict__[args.arch]()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if args.evaluate:
        all_preds = []
        all_targets_binary = []
        all_preds_binary = []
        accs, precisions, recalls, f1s, roc_aucs, sensitivities, specificities = [], [], [], [], [], [], []
        weight_vals = []

        for i, ckp_path in enumerate(args.test_ckp):
            ETdata_test = folds[i]
            test_subset = Subset(full_dataset, ETdata_test)
            testloader = DataLoader(test_subset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            checkpoint = torch.load(ckp_path, map_location='cuda:0')
            model.load_state_dict(checkpoint['state_dict'])
            print('\nEvaluation only for checkpoint:', ckp_path)
            acc, precision, recall, f1, roc_auc, sensitivity, specificity, targets_binary, preds_binary, preds = test(testloader, model, use_cuda)
            all_preds.extend(preds)
            all_targets_binary.extend(targets_binary)
            all_preds_binary.extend(preds_binary)
            accs.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            roc_aucs.append(roc_auc)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            # weight_vals.append(weight_mean)

        # Calculate mean and std for each metric
        acc_mean, acc_std = np.mean(accs), np.std(accs)

        precision_mean, precision_std = np.mean(precisions), np.std(precisions)
        recall_mean, recall_std = np.mean(recalls), np.std(recalls)
        f1_mean, f1_std = np.mean(f1s), np.std(f1s)
        roc_auc_mean, roc_auc_std = np.mean(roc_aucs), np.std(roc_aucs)
        sensitivity_mean, sensitivity_std = np.mean(sensitivities), np.std(sensitivities)
        specificity_mean, specificity_std = np.mean(specificities), np.std(specificities)
        z = 1.96
        n_per_fold = 10
        sensitivity_ci = (sensitivity_mean - z * (sensitivity_std / np.sqrt(n_per_fold)),sensitivity_mean + z * (sensitivity_std / np.sqrt(n_per_fold)))
        specificity_ci = (specificity_mean - z * (specificity_std / np.sqrt(n_per_fold)),specificity_mean + z * (specificity_std / np.sqrt(n_per_fold)))
        # weight_cat = np.concatenate(weight_vals, axis=0)
        # print(weight_cat.shape)
        # weight_mean = np.mean(weight_cat, axis=0)
        # weight_split = np.split(weight_cat, 4, axis=1)
        # print(f'Weight Mean: {weight_mean}')

        # print(f"Accuracy: Mean = {acc_mean}, Std = {acc_std}")
        # # print(f"Precision: Mean = {precision_mean}, Std = {precision_std}")
        # # print(f"Recall: Mean = {recall_mean}, Std = {recall_std}")
        # # print(f"F1 Score: Mean = {f1_mean}, Std = {f1_std}")
        # print(f"ROC-AUC: Mean = {roc_auc_mean}, Std = {roc_auc_std}")
        # print(f"Sensitivity: Mean = {sensitivity_mean}, Std = {sensitivity_std}", f"95% CI = ({sensitivity_ci[0]:.4f}, {sensitivity_ci[1]:.4f})")
        # print(f"Specificity: Mean = {specificity_mean}, Std = {specificity_std}", f"95% CI = ({specificity_ci[0]:.4f}, {specificity_ci[1]:.4f})")
        print(f"Accuracy: {acc_mean:.2f}±{acc_std:.2f}")
        print(f"ROC-AUC: {roc_auc_mean:.2f}±{roc_auc_std:.2f}")
        print(f"Precision: {precision_mean:.2f}±{precision_std:.2f}")
        print(f"Recall: {recall_mean:.2f}±{recall_std:.2f}")
        print(f"F1 Score: {f1_mean:.2f}±{f1_std:.2f}")
        print(f"Sensitivity: {sensitivity_mean:.2f}±{sensitivity_std:.2f} ({sensitivity_ci[0]:.2f}, {sensitivity_ci[1]:.2f})")
        print(f"Specificity: {specificity_mean:.2f}±{specificity_std:.2f} ({specificity_ci[0]:.2f}, {specificity_ci[1]:.2f})")

        save_root = os.path.join('experiments/adncgcls', args.arch)
        if save_root:
            if not os.path.exists(save_root):
                os.makedirs(save_root)
        try:
            all_preds_df = pd.read_csv('experiments/adncgcls/adncgcls.csv')
        except FileNotFoundError:
            all_preds_df = pd.DataFrame()
        
        all_preds_df[args.arch] = all_preds
        all_preds_df.to_csv('experiments/adncgcls/adncgcls.csv', index=False)

        # ## save evaluation results
        # if save_root:
        #     if not os.path.exists(save_root):
        #         os.makedirs(save_root)
        # try:
        #     all_df = pd.read_csv('experiments/ablation/Indicator.csv')
        # except FileNotFoundError:
        #     all_df = pd.DataFrame()
        
        # all_df[args.arch + '_auc'] = roc_aucs
        # all_df.to_csv('experiments/ablation/Indicator.csv', index=False)
        # all_df[args.arch + '_sens'] = sensitivities
        # all_df.to_csv('experiments/ablation/Indicator.csv', index=False)
        # all_df[args.arch + '_spec'] = specificities
        # all_df.to_csv('experiments/ablation/Indicator.csv', index=False)
        ## save evaluation results

        boxplot2(all_preds, all_targets_binary, save_path=save_root)
        violinplot2(all_preds, all_targets_binary, save_path=save_root)
        plot_roc(all_targets_binary, all_preds, roc_auc_mean, args.aucname,  save_path=save_root)


def test(testloader, model, use_cuda):
    model.eval()
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    preds_all = []
    targets_all = []
    weights_all = []
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
            if args.arch == 'biformer' or args.arch == 'transnext' or args.arch == 'rmt' :
                outputs = model(heatmaps)
                preds = torch.sigmoid(outputs)
            elif args.arch == 'mcgrl':
                _, _, _, outputs = model(heatmaps)
                preds = torch.sigmoid(outputs)
            elif args.arch == 'lagmf':
                out, graph_final_out, _ = model(heatmaps, gazegraphs, age_edu, targets)
                _, preds = graph_final_out.max(1)
            elif args.arch == 'dlieg':
                outputs= model(heatmaps, gazegraphs)
                preds = torch.sigmoid(outputs)
            else:
                outputs, _ = model(heatmaps, gazegraphs, age_edu)
                preds = torch.sigmoid(outputs)

            preds_all.append(preds.detach().cpu().numpy())
            targets_all.append(targets.detach().cpu().numpy())  
            # weights_all.append(weights.detach().cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - start_time)

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        )
            bar.next()
        bar.finish()
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
    # sensitivity_se = np.sqrt((sensitivity * (1 - sensitivity)) / len(targets_binary))
    # specificity_se = np.sqrt((specificity * (1 - specificity)) / len(targets_binary))
    # sensitivity_conf_int = stats.norm.interval(0.95, loc=sensitivity, scale=sensitivity_se)
    # specificity_conf_int = stats.norm.interval(0.95, loc=specificity, scale=specificity_se)

    # print('Sensitivity 95% Confidence Interval: {:.4f} to {:.4f}'.format(sensitivity_conf_int[0], sensitivity_conf_int[1]))
    # print('Specificity 95% Confidence Interval: {:.4f} to {:.4f}'.format(specificity_conf_int[0], specificity_conf_int[1]))
    print('Total Time: {:.3f}s | ACC: {:.4f} | Sensitivity: {:.4f} | Specificity: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f} | ROC-AUC: {:.4f}'.format(time.time() - start_time, acc, sensitivity, specificity, precision, recall, f1, roc_auc))
    # Save ROC curve image

    return acc, precision, recall, f1, roc_auc, sensitivity, specificity, targets_binary, preds_binary, np.concatenate(preds_all)


def cal_sens_spec(targets, preds):
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()

    # Calculate sensitivity (recall) and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

if __name__ == '__main__':
    main()
