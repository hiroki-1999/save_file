import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
import numpy as np
import torch.optim as optim
from collections import defaultdict
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from modules.byol import *
from modules.simclr import *
from resnet import ResNet18

parser = argparse.ArgumentParser()
parser.add_argument("--image_size", default=32, type=int, help="Image size")
parser.add_argument(
    "--learning_rate", default=3e-4, type=float, help="Initial learning rate."
)
parser.add_argument(
    "--batch_size", default=256, type=int, help="Batch size for training."
)
parser.add_argument(
    "--num_epochs", default=1000, type=int, help="Number of epochs to train for."
)
parser.add_argument(
    "--resnet_version", default="resnet18", type=str, help="ResNet version."
)
parser.add_argument(
    "--checkpoint_epochs",
    default=1,
    type=int,
    help="Number of epochs between checkpoints/summaries.",
)
parser.add_argument(
    "--dataset_dir",
    default="./datasets",
    type=str,
    help="Directory where dataset is stored.",
)
parser.add_argument(
    "--num_workers",
    default=4,
    type=int,
    help="Number of data loading workers (caution with nodes!)",
)
parser.add_argument(
    "--nodes", default=1, type=int, help="Number of nodes",
)
parser.add_argument("--gpus", default=1, type=int, help="number of gpus per node")
parser.add_argument("--nr", default=0, type=int, help="ranking within the nodes")

# colab work-around
args = parser.parse_args(args=[])

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(0)

# dataset
train_dataset = CIFAR10(root='./data', train=True,download=True, transform=TransformsSimCLR(size=args.image_size))
#train_dataset = datasets.CIFAR100(
#    args.dataset_dir,
#    download=True,
#    transform=TransformsSimCLR(size=args.image_size), # paper 224
#)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    drop_last=True,
    num_workers=args.num_workers,
)

# model
if args.resnet_version == "resnet18":
    #resnet = models.resnet18(pretrained=False)
    resnet = ResNet18(num_classes=10)
elif args.resnet_version == "resnet50":
    resnet = models.resnet50(pretrained=False)
else:
    raise NotImplementedError("ResNet not implemented")

#model = BYOL(resnet, image_size=args.image_size, hidden_layer="avgpool")


#使うもの次第で変更
#これは使わない

model = BYOL(resnet, image_size=args.image_size, hidden_layer="")
model = model.to(device)

#毎回変更必須
main_folder = './save_BYOL'
file_name = 'c10def_400'
file_ver  = '_ver1'
weight_folder = 'weight'
log_folder = 'log'
save_folder =  file_name + file_ver
os.makedirs(main_folder, exist_ok = True)
os.makedirs(main_folder + '/' + save_folder, exist_ok = True)
os.makedirs(main_folder + '/' + save_folder+ '/' + weight_folder, exist_ok = True)
os.makedirs(main_folder + '/' + save_folder+ '/' + log_folder, exist_ok = True)
PATH_save_log     = main_folder + '/' + save_folder +  '/' +log_folder + '/' + file_name + file_ver + '.txt'
PATH_save_log_acc = main_folder + '/' + save_folder +  '/' +log_folder + '/' + file_name + '_acc'+ file_ver + '.txt'
A = ["------------------------------------------"]
np.savetxt(PATH_save_log , A , fmt='%s',newline=', \n')
np.savetxt(PATH_save_log_acc , A , fmt='%s',newline=', \n')
A = [file_name]
with open(PATH_save_log , 'a') as f_handle:
    np.savetxt(f_handle, A , fmt='%s',newline = '\n')
    f_handle.close()
A = [file_name+'acc']
with open(PATH_save_log_acc , 'a') as f_handle:
    np.savetxt(f_handle, A , fmt='%s',newline = '\n')
    f_handle.close()
#model.load_state_dict(torch.load(PATH))
#resnet.load_state_dict(torch.load(PATH))
ACC = []
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
#optimizer.load_state_dict(torch.load(PATH))

# solver
global_step = 0
for epoch in range(args.num_epochs):
    metrics = defaultdict(list)
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        x_i = x_i.to(device)
        x_j = x_j.to(device)

        loss = model(x_i, x_j)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_moving_average()  # update moving average of target encoder

        if step % 10 == 0:
            print(f"Step [{step}/{len(train_loader)}]:\tLoss: {loss.item()}")
            A = [(f"Step [{step}/{len(train_loader)}]:\tLoss: {loss.item()}")]
            with open(PATH_save_log , 'a') as f_handle:
                np.savetxt(f_handle, A , fmt='%s')
                f_handle.close()

        metrics["Loss/train"].append(loss.item())
        global_step += 1

    # write metrics to TensorBoard
    A = [(f"Epoch [{epoch}/{args.num_epochs}]: " + "\t".join([f"{k}: {np.array(v).mean()}" for k, v in metrics.items()]))]
    with open(PATH_save_log , 'a') as f_handle:
                np.savetxt(f_handle, A , fmt='%s')
                f_handle.close()
    with open(PATH_save_log_acc , 'a') as f_handle:
                np.savetxt(f_handle, A , fmt='%s')
                f_handle.close()
    print(f"Epoch [{epoch}/{args.num_epochs}]: " + "\t".join([f"{k}: {np.array(v).mean()}" for k, v in metrics.items()]))
    A = [np.array(v).mean() for k, v in metrics.items()]
    b = A[0]
    ACC.append(b)
    if (epoch + 1) % 10 == 0: 
        PATH = main_folder + '/' + save_folder + '/' + weight_folder + '/' + file_name + '_' + 'B_res' + file_ver + '_epoch_' + str(epoch+1) + '.pth'
        torch.save(resnet.state_dict(), PATH)
        PATH = main_folder + '/' + save_folder + '/' + weight_folder + '/' + file_name + '_' + 'B_opt' + file_ver + '_epoch_' + str(epoch+1) + '.pth'
        torch.save(optimizer.state_dict(), PATH)
        PATH = main_folder + '/' + save_folder + '/' + weight_folder + '/' + file_name + '_' + 'B'     + file_ver + '_epoch_' + str(epoch+1) + '.pth'
        torch.save(model.state_dict(), PATH)
    

    #if epoch % args.checkpoint_epochs == 0:
    #    print(f"Saving model at epoch {epoch}")
    #    torch.save(resnet.state_dict(), f"./model-{epoch}.pt")

with open(PATH_save_log_acc , 'a') as f_handle:
                np.savetxt(f_handle, ACC , fmt='%s')
                f_handle.close()
print(ACC)
# save your improved network
torch.save(resnet.state_dict(), main_folder + '/' + save_folder + "/model-final.pt")