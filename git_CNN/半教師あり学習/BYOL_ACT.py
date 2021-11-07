import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models#, datasets
from torchvision.datasets import CIFAR100
import numpy as np
from collections import defaultdict
# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
#from modules.byol import BYOL
#from modules.simclr import TransformsSimCLR
import torch.nn.functional as F
#from torchsummary import summary


def BYOL_ACT(args,model,resnet,optimizer,train_loader,Q):
    LOSS=[]
    print("START")
    global_step = 0
    for epoch in range(args.start_epochs_BYOL,args.num_epochs_BYOL):
        metrics = defaultdict(list)
        for step, ((x_i, x_j), _) in enumerate(train_loader):
            x_i = x_i.to(args.device)
            x_j = x_j.to(args.device)

            loss = model(x_i, x_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_moving_average()  # update moving average of target encoder

            if step % 10 == 0:
                print(f"Step [{step}/{len(train_loader)}]:\tLoss: {loss.item()}")
                A=[(f"Step [{step}/{len(train_loader)}]:\tLoss: {loss.item()}\n")]
                with open('./'+args.save_file+'/'+args.save_log_title+'_log.txt', 'a') as f_handle:
                    np.savetxt(f_handle, A, fmt='%s',newline=', ')
                    f_handle.close()
            LOSS.append(loss.item())
            metrics["Loss/train"].append(loss.item())
            global_step += 1

        # write metrics to TensorBoard
        print(f"Epoch [{epoch}/{args.num_epochs_BYOL}]: " + "\t".join([f"{k}: {np.array(v).mean()}" for k, v in metrics.items()]))
        A=[(f"Epoch [{epoch}/{args.num_epochs_BYOL}]: " + "\t".join([f"{k}: {np.array(v).mean()}\n" for k, v in metrics.items()]))]
        with open('./'+args.save_file+'/'+args.save_log_title+'_log.txt', 'a') as f_handle:
                    np.savetxt(f_handle, A, fmt='%s',newline=', ')
                    f_handle.close()

        PATH = './'+ args.save_file +'/'+ args.save_log_title + args.save_BYOL_res + 'ver'+ str(Q+1) +'.pth'
        torch.save(resnet.state_dict(), PATH)
        PATH = './'+ args.save_file +'/c10_opt_ver1.pth'
        torch.save(optimizer.state_dict(), PATH)
        PATH = './'+ args.save_file +'/'+ args.save_log_title + args.save_BYOL + 'ver'+ str(Q+1) +'.pth'
        torch.save(model.state_dict(), PATH)

        #if epoch % args.checkpoint_epochs == 0:
        #    print(f"Saving model at epoch {epoch}")
        #    torch.save(resnet.state_dict(), f"./model-{epoch}.pt")

    # save your improved network
    torch.save(resnet.state_dict(), "./"+ args.save_file +'/' + "model-final.pt")
    return LOSS

    
