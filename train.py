import os
import sys
import time
import argparse
import pickle
import glob

# Parse GPU selection before importing torch
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', default="0", type=str, help='GPU device number')
parser.add_argument('--obs_len', type=int, default=10)
parser.add_argument('--pred_len', type=int, default=10)
parser.add_argument('--dataset', default='eth',
                    help='eth,hotel,univ,zara1,zara2')
parser.add_argument('--batch_size', type=int, default=32,
                    help='minibatch size (used for gradient accumulation)')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of epochs')
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gradient clipping')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='learning rate')
parser.add_argument('--milestones', type=int, default=[0, 100],
                    help='number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=False,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='AddGCN_10_10', help='personal tag for the model')
parser.add_argument('--resume', action="store_true", default=False,
                    help='Resume training from last checkpoint')

# Parse early to set CUDA device before importing torch
args_early, _ = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args_early.gpu_num
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from metrics import *
from model import *
from utils import *
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

print("Training initiating....")
print(args)


class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()  # FIX: flush ώστε τα logs να γράφονται αμέσως στο αρχείο


def huber_loss(pred, target, delta=1.0):
    """Huber loss — robust to outliers, used by DualSTMA."""
    return F.huber_loss(pred, target, delta=delta, reduction='mean')


def graph_loss(V_pred, V_target, last_obs):
    """
    DualSTMA-aligned loss (Section 3.4):
      L = 10 * L_position + 0.1 * L_velocity

    Args:
        V_pred:   [pred_len, N, 2] — predicted (lon_vel, lat_vel)
        V_target: [pred_len, N, 4] — GT velocities (first 2 = lon_vel, lat_vel)
        last_obs: [N, 2]           — last observed absolute position (norm space)
    """
    pred_len = V_pred.shape[0]

    # Velocity loss
    vel_loss = huber_loss(V_pred, V_target[:, :, :2])

    # Position loss: cumsum of velocities → absolute positions
    pred_pos = torch.cumsum(V_pred, dim=0) + last_obs.unsqueeze(0)
    gt_pos   = torch.cumsum(V_target[:, :, :2], dim=0) + last_obs.unsqueeze(0)

    # Short-term weighting: step 0 (10min) → highest weight
    weights = torch.tensor(
        [(pred_len - t) for t in range(pred_len)],
        dtype=torch.float32, device=V_pred.device
    )
    weights = weights / weights.sum()

    pos_loss = sum(
        weights[t] * huber_loss(pred_pos[t], gt_pos[t])
        for t in range(pred_len)
    )

    return 10.0 * pos_loss + 0.1 * vel_loss


def make_identity(T, N, device):
    identity_spatial  = torch.ones((T, N, N), device=device) * torch.eye(N, device=device)
    identity_temporal = torch.ones((N, T, T), device=device) * torch.eye(T, device=device)
    return [identity_spatial, identity_temporal]


metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {
    'min_val_epoch': -1, 'min_val_loss': 9999999999999999,
    'min_train_epoch': -1, 'min_train_loss': 9999999999999999
}


def train(epoch, model, optimizer, checkpoint_dir, loader_train, scaler):
    global metrics, constant_metrics
    model.train()

    loss_batch    = 0
    batch_count   = 0
    backward_count = 0
    is_fst_loss   = True
    loss          = None   # FIX: ρητή αρχικοποίηση σε None
    loader_len    = len(loader_train)
    turn_point    = int(loader_len / args.batch_size) * args.batch_size + \
                    loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_train):
        batch_count += 1
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, V_tr = batch

        T = V_obs.shape[1]
        N = V_obs.shape[2]
        identity = make_identity(T, N, device)

        with autocast():
            V_pred   = model(V_obs, identity)
            V_pred   = V_pred.squeeze(0) if V_pred.dim() == 4 else V_pred
            V_target = V_tr.squeeze(0)
            last_obs = obs_traj.squeeze(0)[:, :2, -1]

            l = graph_loss(V_pred, V_target, last_obs)

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            # Accumulation phase: συσσωρεύουμε loss
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss = loss + l
            # FIX: διαγράφουμε το l αμέσως — δεν χρειάζεται πλέον
            del l

        else:
            # Backward phase: κάνουμε το backward pass
            if is_fst_loss:
                loss = l
            else:
                loss = loss + l
            del l  # FIX: διαγράφουμε το l

            loss = loss / args.batch_size
            is_fst_loss = True

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            if args.clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            scaler.step(optimizer)
            scaler.update()

            backward_count += 1
            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / backward_count)

            # FIX: καθαρισμός GPU memory μετά από κάθε backward pass
            del loss, V_pred, V_target, last_obs, identity
            loss = None  # reset για τον επόμενο accumulation κύκλο
            torch.cuda.empty_cache()

    metrics['train_loss'].append(loss_batch / max(1, backward_count))

    if metrics['train_loss'][-1] < constant_metrics['min_train_loss']:
        constant_metrics['min_train_loss'] = metrics['train_loss'][-1]
        constant_metrics['min_train_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'train_best.pth')

    # Save last checkpoint for resume support
    torch.save(model.state_dict(), checkpoint_dir + 'last.pth')
    with open(checkpoint_dir + 'last_epoch.txt', 'w') as f:
        f.write(str(epoch))


def vald(epoch, model, checkpoint_dir, loader_val):
    global metrics, constant_metrics
    model.eval()

    loss_batch     = 0
    batch_count    = 0
    backward_count = 0
    is_fst_loss    = True
    loss           = None  # FIX: ρητή αρχικοποίηση σε None
    loader_len     = len(loader_val)
    turn_point     = int(loader_len / args.batch_size) * args.batch_size + \
                     loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_val):
        batch_count += 1
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, V_tr = batch

        with torch.no_grad():
            T = V_obs.shape[1]
            N = V_obs.shape[2]
            identity = make_identity(T, N, device)

            with autocast():
                V_pred   = model(V_obs, identity)
                V_pred   = V_pred.squeeze(0) if V_pred.dim() == 4 else V_pred
                V_target = V_tr.squeeze(0)
                last_obs = obs_traj.squeeze(0)[:, :2, -1]

                l = graph_loss(V_pred, V_target, last_obs)

            if batch_count % args.batch_size != 0 and cnt != turn_point:
                if is_fst_loss:
                    loss = l
                    is_fst_loss = False
                else:
                    loss = loss + l
                del l  # FIX: αποδέσμευση αμέσως

            else:
                if is_fst_loss:
                    loss = l
                else:
                    loss = loss + l
                del l  # FIX: αποδέσμευση αμέσως

                loss = loss / args.batch_size
                is_fst_loss = True
                backward_count += 1
                loss_batch += loss.item()
                print('VALD:', '\t Epoch:', epoch, '\t Loss:', loss_batch / backward_count)

                # FIX: καθαρισμός GPU memory — στο val δεν έχουμε gradients
                # αλλά τα tensors κρατούν GPU memory ακόμα
                del loss, V_pred, V_target, last_obs, identity
                loss = None  # reset για τον επόμενο accumulation κύκλο
                torch.cuda.empty_cache()

    avg_loss = loss_batch / max(1, backward_count)
    metrics['val_loss'].append(avg_loss)
    print('VALD:', '\t Epoch:', epoch, '\t Loss:', avg_loss)

    if avg_loss < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = avg_loss
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')


def main(args):
    obs_seq_len  = args.obs_len
    pred_seq_len = args.pred_len

    data_set = os.path.join('./dataset', args.dataset)

    def _get_split_csv_files(split_name):
        split_dir = os.path.join(data_set, split_name)
        csv_files = sorted(glob.glob(os.path.join(split_dir, '*.csv')))
        return split_dir, csv_files

    train_dir, train_csv_files = _get_split_csv_files('train')
    if not train_csv_files:
        raise RuntimeError(
            f"No CSV files found in {train_dir}. "
            f"Run: python convert_json_to_csv.py"
        )

    val_dir, val_csv_files = _get_split_csv_files('val')
    if not val_csv_files:
        raise RuntimeError(
            f"No CSV files found in {val_dir}. "
            f"Run: python convert_json_to_csv.py"
        )

    dset_train = TrajectoryDataset(
        os.path.join(data_set, 'train'),
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_train = DataLoader(
        dset_train,
        batch_size=1,
        shuffle=True,
        num_workers=0)  # FIX: 0 workers — αποφεύγει shared memory exhaustion στον DGX

    dset_val = TrajectoryDataset(
        os.path.join(data_set, 'val'),
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_val = DataLoader(
        dset_val,
        batch_size=1,
        shuffle=False,
        num_workers=0)  # FIX: 0 workers — αποφεύγει shared memory exhaustion στον DGX

    print('Training started ...')
    print(f'Using device: {device}')

    writer = SummaryWriter(f"runs/{args.tag}_{args.dataset}_{time.strftime('%Y%m%d-%H%M%S')}")

    model = TrajectoryModel(
        number_asymmetric_conv_layer=2,
        embedding_dims=64,
        number_gcn_layers=1,
        dropout=0,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        out_dims=2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler    = GradScaler()

    checkpoint_dir = './checkpoints/' + args.tag + '/' + args.dataset + '/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Resume support
    start_epoch = 0
    if args.resume:
        last_ckpt          = checkpoint_dir + 'last.pth'
        last_epoch_file    = checkpoint_dir + 'last_epoch.txt'
        constant_metrics_file = checkpoint_dir + 'constant_metrics.pkl'
        if os.path.exists(last_ckpt) and os.path.exists(last_epoch_file):
            model.load_state_dict(torch.load(last_ckpt, map_location=device))
            with open(last_epoch_file) as f:
                start_epoch = int(f.read()) + 1
            if os.path.exists(constant_metrics_file):
                with open(constant_metrics_file, 'rb') as fp:
                    constant_metrics.update(pickle.load(fp))
            print(f'Resumed from epoch {start_epoch}')
        else:
            print('No checkpoint found, starting from scratch')

    if args.use_lrschd:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[0, 100], gamma=0.1
        )

    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    for epoch in range(start_epoch, args.num_epochs):
        train(epoch, model, optimizer, checkpoint_dir, loader_train, scaler)
        vald(epoch, model, checkpoint_dir, loader_val)

        writer.add_scalar('trainloss', np.array(metrics['train_loss'])[-1], epoch)
        writer.add_scalar('valloss',   np.array(metrics['val_loss'])[-1],   epoch)

        if args.use_lrschd:
            scheduler.step()

        print('*' * 30)
        print('Epoch:', args.dataset + '/' + args.tag, ":", epoch)
        for k, v in metrics.items():
            if len(v) > 0:
                print(k, v[-1])

        print(constant_metrics)
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