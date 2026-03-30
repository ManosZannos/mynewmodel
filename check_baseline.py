import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import TrajectoryDataset

dataset = TrajectoryDataset('dataset/marinecadastre_2021/test', obs_len=10, pred_len=5, skip=1)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

LAT_MIN, LAT_RANGE = 20.90883, 28.32044
LON_MIN, LON_RANGE = -133.29703, 72.60811

errors_deg = []

for i, batch in enumerate(loader):
    if i >= 1000: break
    obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, _, _, V_obs, V_tr = batch

    last_obs = obs_traj.squeeze(0)[:, :2, -1]

    gt_abs = pred_traj_gt.squeeze(0).permute(2,0,1)[:,:,:2]
    gt_lon = gt_abs[:,:,0] * LON_RANGE + LON_MIN
    gt_lat = gt_abs[:,:,1] * LAT_RANGE + LAT_MIN
    zero_lon = last_obs[:,0].unsqueeze(0).expand(5,-1) * LON_RANGE + LON_MIN
    zero_lat = last_obs[:,1].unsqueeze(0).expand(5,-1) * LAT_RANGE + LAT_MIN

    zero_error = torch.sqrt((zero_lon-gt_lon)**2 + (zero_lat-gt_lat)**2).mean().item()
    errors_deg.append(zero_error)

print(f'Zero-velocity baseline ADE: {np.mean(errors_deg):.6f}°')
print(f'Our model ADE:              0.037901°')
print(f'DualSTMA ADE:               0.002436°')
