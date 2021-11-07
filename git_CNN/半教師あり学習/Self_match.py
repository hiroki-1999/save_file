#region
import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100  
from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy
from Fixmatch_ACT import *
from resnet import ResNet18
import torch.nn as nn
#endregion
#region

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from modules.byol import *
from modules.simclr import *
from torch.utils.tensorboard import SummaryWriter

from BYOL_ACT import *

from torchvision import models
#from torchvision.datasets import CIFAR100

# distributed training

#endregion
#region

#endregion
best_acc = 0
def main(FLAG):
    #region
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    #BYOL 
    parser.add_argument("--image_size", default=32, type=int, help="Image size")
    parser.add_argument(
    "--num_workers",default=4,type=int,help="Number of data loading workers (caution with nodes!)")
    #endregion
    parser.add_argument('--dataset', default='cifar100', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--seed', default=1, type=int,
                        help="random seed")
    parser.add_argument('--num-labeled', type=int, default=10000,
                        help='number of labeled data')
    #EPOCH
    parser.add_argument('--batch_size', default=32, type=int,
                        help='train batchsize')
    parser.add_argument('--epochs', default=150, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--eval-step', default=1563, type=int,
                        help='number of eval steps to run')
    #parser.add_argument('--eval-step', default=150, type=int,
    #                     help='number of eval steps to run')
    parser.add_argument('--total-steps', default=150*1563, type=int,
                        help='number of total steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    #BYOL EPOCH
    parser.add_argument("--num_epochs_BYOL", default=400, type=int, 
                        help="Number of epochs to train for.")
    parser.add_argument("--start_epochs_BYOL", default=0, type=int, 
                        help="Number of epochs to train for.")
    parser.add_argument("--learning_rate_BYOL", default=3e-4, type=float, 
                        help="Initial learning rate.")
    parser.add_argument('--batch_size_BYOL', default=200, type=int,
                        help='train batchsize')
    #save_point
    parser.add_argument('--out', default='c10_result',
                        help='directory to output the result')
    parser.add_argument('--save_Fix', default='_Fix_', type=str,#v1.pth.tar
                        help='save-name')
    # parser.add_argument('--save_Fix_make', default='result/c10_v1.pth.tar', type=str,
    #                     help='save-name')
    parser.add_argument('--save_BYOL', default='_B_', type=str,
                        help='save-name')
    parser.add_argument('--save_BYOL_res', default='_c10_res_', type=str,
                        help='save-name')
    
    parser.add_argument('--save_file', default='save_22', type=str,
                        help='save-file')
    parser.add_argument('--save_file_sub', default='10_07', type=str,
                        help='save-file')
    parser.add_argument('--save_log_title_b', default='c100_L10000_150_400', type=str,
                        help='save-file')

    #PATH Load
    parser.add_argument('--flag', default=0, type=int,#FLAG
                        help='flag 0:normal , 1:Fixmatch_Path start , 2:BYOL_Path start')
    #Fixmatch Path FLAG = 1
    parser.add_argument('--resume', default=None, type=str,##None #Fixmatch Path
                        help='path to latest checkpoint (default: none)')
    #BYOL Path     FLAG = 2
    parser.add_argument('--PATH_name', default=none, type=str,
                        help='save-file')

    #parser.add_argument('--save_log_title', default='c100_L10000_150_400_vZZZ', type=str,
    #                    help='save-file')
    parser.add_argument('--save_log_title', default='c100_L10000_20%_150_400_v2', type=str,
                        help='save-file')
    #num run
    parser.add_argument('--run', default=1, type=int,
                        help='run')

    args = parser.parse_args(args=[])
    global best_acc
    FLAG = args.flag
    #region
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    ### Fix Dataset
    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    
    ### BYOL Dataset
    if args.dataset == 'cifar10' :
        train_dataset = CIFAR10(root='../torch-datasets', train=True,download=True, transform=TransformsSimCLR(size=args.image_size))
    elif args.dataset == 'cifar100':
        train_dataset = CIFAR100(root='../torch-datasets', train=True,download=True, transform=TransformsSimCLR(size=args.image_size))
    
    BYOL_train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size_BYOL,
    drop_last=True,
    num_workers=args.num_workers,
    )
    #endregion
    Resnet = create_model(args,logger)
    if args.local_rank == 0:
        torch.distributed.barrier()
    Resnet.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in Resnet.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in Resnet.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, Resnet, args.ema_decay)

    args.start_epoch = 0

    #region
    TRAIN_FIX_1=[]
    TRAIN_FIX_5=[]
    TRAIN_FIX_BEST=[]
    TRAIN_FIX_MEAN=[]
    TEST_FIX=[]
    LOSS_BYOL=[]
    ABCD=['\n','NOW']
    EFGH=['\n','NOW','\n']
    if args.local_rank in [-1, 0]:
        os.makedirs(args.save_file, exist_ok=True)
        #args.writer = SummaryWriter(args.out)
    
    np.savetxt('./'+args.save_file+'/'+args.save_log_title+'_log.txt', EFGH, fmt='%s',newline=', ')

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)#checkpoint.pth.tar
        #args.out = os.path.dirname('./checkpoint2.pth.tar')
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        #args.start_epoch = checkpoint['epoch']
        #Resnet.load_state_dict(checkpoint['state_dict'])
        Resnet.load_state_dict(checkpoint['ema_state_dict'])
        # if args.use_ema:
        #     ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        EFGH=['\n','load ok','\n']
        with open('./'+args.save_file+'/'+args.save_log_title+'_log.txt', 'a') as f_handle:
            np.savetxt(f_handle,EFGH , fmt='%s',newline=', ')
            f_handle.close()
        print("load FLAG1 ok")
    #endregion
    #region
    if args.amp:
        from apex import amp
        Resnet, optimizer = amp.initialize(
            Resnet, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        Resnet = torch.nn.parallel.DistributedDataParallel(
            Resnet, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")
    RESNET = ResNet18(num_classes=args.num_classes)
    optimizer_Adam = torch.optim.Adam(RESNET.parameters(), lr=args.learning_rate_BYOL)
    Resnet.zero_grad()
    
    #endregion
    #device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    
    
    if FLAG == 0:
        PATH = './' + args.PATH_name
        Resnet.load_state_dict(torch.load(PATH))
    # if FLAG == 1:
    #     PATH = './'+args.PATH_name
    #     checkpoint = torch.load(PATH)    
    #     #Resnet.load_state_dict(checkpoint['state_dict'])
    #     Resnet.load_state_dict(checkpoint['ema_state_dict'])
    #     ema_model = ModelEMA(args, Resnet, args.ema_decay)
    #     #ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
    
    for Q in range(args.run):
        #Fixmatchの実行
        if FLAG != 1:
            print("FLAG != 1 ok")
            TRAIN_FIX_A , TRAIN_FIX_B, TRAIN_FIX_C, TRAIN_FIX_D=train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
                Resnet, optimizer, ema_model, scheduler,logger,Q)
            TRAIN_FIX_1.append(TRAIN_FIX_A)
            TRAIN_FIX_5.append(TRAIN_FIX_B)
            TRAIN_FIX_BEST.append(TRAIN_FIX_C)
            TRAIN_FIX_MEAN.append(TRAIN_FIX_D)
            
        elif Q != 0:
            print("Fix Q != 0 ok")
            TRAIN_FIX_A , TRAIN_FIX_B, TRAIN_FIX_C, TRAIN_FIX_D=train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
                Resnet, optimizer, ema_model, scheduler,logger,Q)
            TRAIN_FIX_1.append(TRAIN_FIX_A)
            TRAIN_FIX_5.append(TRAIN_FIX_B)
            TRAIN_FIX_BEST.append(TRAIN_FIX_C)
            TRAIN_FIX_MEAN.append(TRAIN_FIX_D)
            

        RESNET = ResNet18(num_classes=args.num_classes)
        if (FLAG != 2) & (Q == 0):#ema_modelの採用
            PATH = './' + args.out + '/' + args.save_log_title + args.save_Fix+'ver'+ str(Q+1)+'.pth.tar'
            checkpoint = torch.load(PATH)
            RESNET.load_state_dict(checkpoint['ema_state_dict'])
            print("load Fix_match to B ok")
        if (FLAG == 2) & (Q ==0):
            with torch.no_grad():
                #初回のBYOLの読み込み　BYOLの全結合層
                Resnet = ResNet18(num_classes=args.num_classes)
                PATH = './' + args.PATH_name
                #PATH = './'+ args.out +'/'+ args.save_log_title + args.save_BYOL_res + 'ver'+ str(Q) +'.pth'
                Resnet.load_state_dict(torch.load(PATH))
                print("load first_B To B ok")
                FULL_C = list(RESNET.modules())[-1]

            #前回のFixmatchの読み込み　畳み込み層の重み
            RESNET = ResNet18(num_classes=args.num_classes)    
            PATH = './' + args.out + '/' + args.save_log_title + args.save_Fix+'ver'+ str(Q+1)+'.pth.tar'
            checkpoint = torch.load(PATH)
            RESNET.load_state_dict(checkpoint['ema_state_dict'])
            print("load Fix+BYOL to B ok")
        elif Q > 0:
            #BYOLの重みの更新
            with torch.no_grad():
                #前回のBYOLの全結合層
                Resnet = ResNet18(num_classes=args.num_classes)
                PATH = './'+ args.out +'/'+ args.save_log_title + args.save_BYOL_res + 'ver'+ str(Q) +'.pth'
                Resnet.load_state_dict(torch.load(PATH))
                FULL_C = list(RESNET.modules())[-1]
            #Fixmatchの畳み込み層の重み
            PATH = './' + args.out + '/' + args.save_log_title + args.save_Fix +'ver'+ str(Q+1)+'.pth.tar'
            checkpoint = torch.load(PATH)
            RESNET = ResNet18(num_classes=args.num_classes)
            RESNET.load_state_dict(checkpoint['ema_state_dict'])
            with torch.no_grad():
                RESNET.linear = FULL_C
            print("load Fixm+BYOL to B ok")
            
        model = BYOL(RESNET, image_size=args.image_size, hidden_layer="")
        optimizer_Adam = torch.optim.Adam(RESNET.parameters(), lr=args.learning_rate_BYOL)
        args.device = torch.device('cuda', args.gpu_id)
        model = model.to(args.device)
        #BYOL 実行
        LOSSS=BYOL_ACT(args,model,RESNET,optimizer_Adam,BYOL_train_loader,Q)
        LOSS_BYOL.append(LOSSS)
        
        #Fixmatchの重み　の定義
        #region
        with torch.no_grad():
            RESNET = ResNet18(num_classes=args.num_classes)
            if (FLAG == 1) & (Q == 0):
                PATH = args.resume
                print("load first_Fix To Fix ok")
            else:
                PATH = './' + args.out + '/' + args.save_log_title + args.save_Fix +'ver'+ str(Q+1)+'.pth.tar'
            checkpoint = torch.load(PATH)
            RESNET.load_state_dict(checkpoint['ema_state_dict'])
            FULL_C = list(RESNET.modules())[-1]
        # Fix match　事前学習 
        Resnet = ResNet18(num_classes=args.num_classes)
        PATH = './'+ args.out +'/'+ args.save_log_title + args.save_BYOL_res + 'ver'+ str(Q+1) +'.pth'
        Resnet.load_state_dict(torch.load(PATH))
        with torch.no_grad():
            Resnet.linear = FULL_C
        print("load BYOL+Fix To Fix ok")
        #endregion

        args.device = torch.device('cuda', args.gpu_id)
        Resnet.to(args.device)
        ema_model = ModelEMA(args, Resnet, args.ema_decay)
        #ログ出力
        #region
        args.start_epoch += args.epochs
        args.epochs      += args.epochs
        args.start_epochs_BYOL +=  args.num_epochs_BYOL
        args.num_epochs_BYOL   +=  args.num_epochs_BYOL 
        #print("args.epoch",args.start_epoch,args.epochs)
        #print("args.epoch",args.start_epochs_BYOL,args.num_epochs_BYOL)
        
        if Q == 0 :
            print("------------------------------------------------")
            np.savetxt(args.save_file+'/top1-acc.txt', TRAIN_FIX_1, fmt='%f',newline=', ')
            np.savetxt(args.save_file+'/top5-acc_acc.txt', TRAIN_FIX_5, fmt='%f',newline=', ')
            np.savetxt(args.save_file+'/Best_acc.txt', TRAIN_FIX_BEST, fmt='%f',newline=', ')
            np.savetxt(args.save_file+'/Fix_Mean.txt', TRAIN_FIX_MEAN, fmt='%f',newline=', ')
        else:
            with open(args.save_file+'/top1-acc.txt', 'a') as f_handle:
                np.savetxt(f_handle,ABCD , fmt='%s')
                np.savetxt(f_handle, TRAIN_FIX_1, fmt='%f',newline=', ')
                f_handle.close()
            with open(args.save_file+'/top5-acc_acc.txt', 'a') as f_handle:
                np.savetxt(f_handle,ABCD , fmt='%s')
                np.savetxt(f_handle, TRAIN_FIX_5, fmt='%f',newline=', ')
                f_handle.close()
            with open(args.save_file+'/Best_acc.txt', 'a') as f_handle:
                np.savetxt(f_handle,ABCD , fmt='%s')
                np.savetxt(f_handle, TRAIN_FIX_BEST, fmt='%f',newline=', ')
                f_handle.close()
            with open(args.save_file+'/Fix_Mean.txt', 'a') as f_handle:
                np.savetxt(f_handle,ABCD , fmt='%s')
                np.savetxt(f_handle, TRAIN_FIX_MEAN, fmt='%f',newline=', ')
                f_handle.close()
        #endregion
    
    #最後のFixmatch
    Q = Q + 1
    TRAIN_FIX_A , TRAIN_FIX_B, TRAIN_FIX_C, TRAIN_FIX_D=train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
            Resnet, optimizer, ema_model, scheduler,logger,Q)
    TRAIN_FIX_1.append(TRAIN_FIX_A)
    TRAIN_FIX_5.append(TRAIN_FIX_B)
    TRAIN_FIX_BEST.append(TRAIN_FIX_C)
    TRAIN_FIX_MEAN.append(TRAIN_FIX_D)
    print('Finished Training')
    print("訓練データの認識率")
    print("訓練データの認識率 -1")
    print(TRAIN_FIX_1)
    print("訓練データの認識率 -5")
    print(TRAIN_FIX_5)
    print("訓練データの認識率 BEST")
    print(TRAIN_FIX_BEST)
    print("訓練データの認識率 MEAN")
    print(TRAIN_FIX_MEAN)

    
    # print("テストデータの認識率")
    # print(TEST_FIX)
    print("BYOL_訓練データのLOSS")
    print(LOSS_BYOL)
    
    with open(args.save_file+'/top1-acc.txt', 'a') as f_handle:
                np.savetxt(f_handle,ABCD , fmt='%s')
                np.savetxt(f_handle, TRAIN_FIX_1, fmt='%f',newline=', ')
                f_handle.close()
    with open(args.save_file+'/top5-acc_acc.txt', 'a') as f_handle:
        np.savetxt(f_handle,ABCD , fmt='%s')
        np.savetxt(f_handle, TRAIN_FIX_5, fmt='%f',newline=', ')
        f_handle.close()
    with open(args.save_file+'/Best_acc.txt', 'a') as f_handle:
        np.savetxt(f_handle,ABCD , fmt='%s')
        np.savetxt(f_handle, TRAIN_FIX_BEST, fmt='%f',newline=', ')
        f_handle.close()
    with open(args.save_file+'/Fix_Mean.txt', 'a') as f_handle:
        np.savetxt(f_handle,ABCD , fmt='%s')
        np.savetxt(f_handle, TRAIN_FIX_MEAN, fmt='%f',newline=', ')
        f_handle.close()

    

if __name__ == '__main__':
    main(FLAG=1)
    #FLAG = 1ならFix飛ばす