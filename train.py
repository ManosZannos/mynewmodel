import os
import sys
import time
import argparse
import pickle
import random

# Parse GPU selection before importing torch
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', default="0", type=str, help='GPU device number')
parser.add_argument('--obs_len', type=int, default=10)
parser.add_argument('--pred_len', type=int, default=10)
parser.add_argument('--dataset', default='eth',
                    help='eth,hotel,univ,zara1,zara2')
parser.add_argument('--batch_size', type=int, default=32,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of epochs')
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='learning rate')
parser.add_argument('--milestones', type=int, default=[0, 100],
                    help='number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=False,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='AddGCN_10_10', help='personal tag for the model ')

# Parse early to set CUDA device before importing torch
args_early, _ = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args_early.gpu_num
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from metrics import *
from model import *
from utils import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()


print("Training initiating....")
print(args)

# 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred, V_target)


metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999, 'min_train_epoch': -1,
                    'min_train_loss': 9999999999999999}


def train(epoch, model, optimizer, checkpoint_dir, loader_train):
    global metrics, constant_metrics
    model.train()
    loss_batch = 0
    num_updates = 0
    loader_len = len(loader_train)
    
    # Accumulation state - clean counters
    accum_loss_sum = 0.0
    accum_samples = 0

    for cnt, batch in enumerate(loader_train):
        # Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr = batch

        # print("obs_len---",obs_traj.shape)[1, 87, 4, 10])

        # obs_traj observed absolute coordinate [1 N 2 obs_len] 
        # pred_traj_gt ground truth absolute coordinate [1 N 2 pred_len]
        # obs_traj_rel velocity of observed trajectory [1 N 2 obs_len]
        # pred_traj_gt_rel velocity of ground-truth [1 N 2 pred_len]
        # non_linear_ped 0/1 tensor indicated whether the trajectory of pedestrians n is linear [1 N]
        # loss_mask 0/1 tensor indicated whether the trajectory point at time t is loss [1 N obs_len+pred_len]
        # V_obs input graph of observed trajectory represented by velocity  [1 obs_len N 3]  ， batch_size ,128
        # V_tr target graph of ground-truth represented by velocity  [1 pred_len N 2]
        # V_tr = V_tr[:,0:2]
        # print("V-TR---",V_tr.shape)

        # PAPER NOTE (Section 3.2.1):
        # According to the paper, adjacency matrices should be initialized as:
        #   - Spatial graph: all elements = 1 (not identity matrix)
        #   - Temporal graph: upper triangular matrix with 1s (not identity matrix)
        # Current implementation uses identity matrices (diagonal=1, rest=0)
        # This may need to be changed to match paper exactly:
        #
        # Correct initialization per paper would be:
    
        #
        # However, these are then filtered by the sparse adjacency generation module,
        # so the current identity matrix approach may be acceptable as a "self-connection" mask.
        
        T = V_obs.shape[1]   # obs_len
        N = V_obs.shape[2]   # num vessels in this sample

        # Paper-faithful initialization (Section 3.2.1):
        # Spatial: all ones (no prior knowledge of vessel interactions)
        identity_spatial = torch.ones((T, N, N), device=device)

        # Temporal: upper triangular matrix with 1s (current state independent of future)
        identity_temporal = torch.triu(torch.ones((N, T, T), device=device), diagonal=0)

        identity = [identity_spatial, identity_temporal]

        # Zero gradients at the start of each accumulation cycle
        if accum_samples == 0:
            optimizer.zero_grad(set_to_none=True)

#V_obs:[128,8,57,3]   identity_spatial：[8,57,57]--- identity_temporal：[57,8,8]
        V_pred = model(V_obs, identity)
        V_pred = V_pred.squeeze(0)  # Remove batch dimension only
        V_tr = V_tr.squeeze(0)      # Remove batch dimension only    

        l = graph_loss(V_pred, V_tr)
        
        # Accumulate unscaled loss for metrics (per-sample)
        accum_loss_sum += l.item()
        accum_samples += 1
        
        # Scale for gradient accumulation
        l = l / args.batch_size
        l.backward()

        # Update weights at the end of each accumulation cycle
        is_last = (cnt == loader_len - 1)
        if accum_samples >= args.batch_size or is_last:
            if args.clip_grad is not None:  
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            
            # Metrics - average per-sample loss for this update
            num_updates += 1
            avg_loss_this_update = accum_loss_sum / accum_samples
            loss_batch += avg_loss_this_update
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / num_updates)
            
            # Reset accumulation counters
            accum_loss_sum = 0.0
            accum_samples = 0
            
    metrics['train_loss'].append(loss_batch / max(1, num_updates))

    if metrics['train_loss'][-1] < constant_metrics['min_train_loss']:   
        constant_metrics['min_train_loss'] = metrics['train_loss'][-1]    
        constant_metrics['min_train_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'train_best.pth')  # OK


def vald(epoch, model, checkpoint_dir, loader_val):
    """
    Validation loop using bivariate Gaussian loss.
    
    NOTE: This computes training loss (negative log-likelihood) for monitoring.
    For paper-aligned evaluation metrics (minADE-20, minFDE-20), use evaluate.py
    which implements best-of-K sampling as described in the paper:
    
    "During the inference stage, 20 samples are extracted from the learned 
    bivariate Gaussian distribution and the closest sample to the ground-truth 
    is used to calculate the performance index of the model."
    
    Usage after training:
        python evaluate.py --checkpoint checkpoints/.../val_best.pth --dataset noaa_dec2021
    """
    global metrics, constant_metrics
    model.eval()
    loss_batch = 0
    batch_count = 0

    for cnt, batch in enumerate(loader_val):
        batch_count += 1
        # Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr = batch

        with torch.no_grad():
            T = V_obs.shape[1]   # obs_len
            N = V_obs.shape[2]   # num vessels in this sample

            # Paper-faithful initialization (Section 3.2.1):
            # Spatial: all ones (no prior knowledge of vessel interactions)
            identity_spatial = torch.ones((T, N, N), device=device)

            # Temporal: upper triangular matrix with 1s (current state independent of future)
            identity_temporal = torch.triu(torch.ones((N, T, T), device=device), diagonal=0)

            identity = [identity_spatial, identity_temporal]

            V_pred = model(V_obs, identity)

            V_pred = V_pred.squeeze(0)  # Remove batch dimension only
            V_tr = V_tr.squeeze(0)      # Remove batch dimension only

            l = graph_loss(V_pred, V_tr)
            loss_batch += l.item()

    avg_loss = loss_batch / max(1, batch_count)
    metrics['val_loss'].append(avg_loss)
    print('VALD:', '\t Epoch:', epoch, '\t Loss:', avg_loss)

    if avg_loss < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = avg_loss
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')  # OK


def main(args):
    obs_seq_len = args.obs_len
    pred_seq_len = args.pred_len 

    data_set = './dataset/' + args.dataset + '/'
    
    dset_train = TrajectoryDataset(
        data_set + 'train/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_train = DataLoader(
        dset_train,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=True,
        num_workers=0)      
    dset_val = TrajectoryDataset(
        data_set + 'val/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_val = DataLoader(
        dset_val,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=False,
        num_workers=0)

    print('Training started ...')
    print(f'Using device: {device}')

    # Initialize TensorBoard writer with unique run directory
    writer = SummaryWriter(f"runs/{args.tag}_{args.dataset}_{time.strftime('%Y%m%d-%H%M%S')}")

    model = TrajectoryModel(number_asymmetric_conv_layer=2, embedding_dims=64, number_gcn_layers=1, dropout=0,
                            obs_len=args.obs_len, pred_len=args.pred_len, out_dims=5).to(device)

    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.use_lrschd:   #true
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0, 100], gamma=0.1)

    # if args.use_lrschd:  #alse
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)

    checkpoint_dir = './checkpoints/' + args.tag + '/' + args.dataset + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    for epoch in range(args.num_epochs):
        train(epoch, model, optimizer, checkpoint_dir, loader_train)    
        vald(epoch, model, checkpoint_dir, loader_val)

        writer.add_scalar('trainloss', np.array(metrics['train_loss'])[epoch], epoch)
        writer.add_scalar('valloss', np.array(metrics['val_loss'])[epoch], epoch)

        if args.use_lrschd:     #true
            scheduler.step()

        print('*' * 30)
        print('Epoch:', args.dataset + '/' + args.tag, ":", epoch)
        for k, v in metrics.items():
            if len(v) > 0:
                print(k, v[-1])    

        print(constant_metrics)   #{'min_val_epoch':  'min_val_loss':   'min_train_epoch':  'min_train_loss':  }
        print('*' * 30)

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)
    
    writer.close()
    print('Training completed!')


if __name__ == '__main__':

    log_path = './Logs_train/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    sys.stdout = Logger(log_file_name)
    sys.stderr = Logger(log_file_name)

    args = parser.parse_args()
    main(args)