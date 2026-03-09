import math
import torch
import numpy as np
#1
def ade(predAll,targetAll,count_):    #[pre_len,N,2]
    All = len(predAll)     #ALL=3
    sum_all = 0 
    for s in range(All):    #0,1,2
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)   
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        
        N = pred.shape[0]
        T = pred.shape[1]     #Tpred
        sum_ = 0 
        for i in range(N):   #0，1
            for t in range(T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)   
        sum_all += sum_/(N*T)
        
    return sum_all/All


def fde(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T-1,T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N)

    return sum_all/All


def seq_to_nodes(seq_,max_nodes = 88):  
    seq_ = seq_.squeeze()           
    seq_ = seq_[:, :2]
    seq_len = seq_.shape[2]         #8
    max_nodes = seq_.shape[0]  

    V = np.zeros((seq_len,max_nodes,2))      #(8,88,2)
    for s in range(seq_len):         #0 1 2 3 4 5 6 7
        step_ = seq_[:,:,s]                   
        for h in range(len(step_)):    
            V[s,h,:] = step_[h]        
            
    return V.squeeze()

def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes = nodes[:, :, :2]
    init_node = init_node[:, :2]
    nodes_ = np.zeros_like(nodes)                   #nodes_: [obs_len N 3]
    for s in range(nodes.shape[0]):          #0-7
        for ped in range(nodes.shape[1]):   
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]     
    return nodes_.squeeze()

def closer_to_zero(current,new_v):
    dec =  min([(abs(current),current),(abs(new_v),new_v)])[1]
    if dec != current:
        return True
    else: 
        return False
        
def bivariate_loss(V_pred,V_trgt):
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_trgt.shape

   # [1 obs_len N 3]
  #  [1 pred_len N 2]    V_trgt:[batch_size,obs_len,num_preds,xy]------[128， 8， 57， 2]
    normx = V_trgt[:,:,0]- V_pred[:,:,0]    
    normy = V_trgt[:,:,1]- V_pred[:,:,1]    

    
    sx = torch.exp(V_pred[:,:,2]) #sx
    sy = torch.exp(V_pred[:,:,3]) #sy
    corr = torch.tanh(V_pred[:,:,4]) #corr
    
    sxsy = sx * sy
    sxsy = torch.clamp(sxsy, min=1e-6)  # Numerical stability

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2
    negRho = torch.clamp(negRho, min=1e-6)  # Numerical stability

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor 
    denom = 2 * torch.pi * (sxsy * torch.sqrt(negRho))  # Use torch.pi for device consistency

    # Final PDF calculation 
    result = result / denom
    # Numerical stability  
    epsilon = 1e-20   
    result = -torch.log(torch.clamp(result, min=epsilon))   
    result = torch.mean(result)    
    
    return result


# ============================================================================
# Best-of-K Sampling Evaluation (Paper-Aligned)
# ============================================================================

def sample_bivariate_gaussian(V_pred, num_samples=20):
    """
    Sample trajectories from bivariate Gaussian distribution.
    
    Paper quote:
    "During the inference stage, 20 samples are extracted from the learned 
    bivariate Gaussian distribution and the closest sample to the ground-truth 
    is used to calculate the performance index of the model."
    
    Args:
        V_pred: Predicted Gaussian parameters [pred_len, N, 5]
                where 5 = [μx, μy, log(σx), log(σy), ρ]
        num_samples: Number of samples to draw (K=20 in paper)
    
    Returns:
        samples: [num_samples, pred_len, N, 2] - sampled (x, y) trajectories
    """
    pred_len, N, _ = V_pred.shape
    device = V_pred.device
    
    # Extract Gaussian parameters
    mux = V_pred[:, :, 0]      # [pred_len, N]
    muy = V_pred[:, :, 1]      # [pred_len, N]
    sx = torch.exp(V_pred[:, :, 2])  # [pred_len, N]
    sy = torch.exp(V_pred[:, :, 3])  # [pred_len, N]
    corr = torch.tanh(V_pred[:, :, 4])  # [pred_len, N]
    
    # Sample from standard bivariate normal: (z1, z2) ~ N(0, I)
    # Shape: [num_samples, pred_len, N, 2]
    z = torch.randn(num_samples, pred_len, N, 2, device=device)
    
    # Transform to correlated bivariate normal using Cholesky decomposition
    # X = μ + L @ Z where L is lower triangular from Cholesky decomposition
    # For 2D: [[sx, 0], [ρ*sy, sy*sqrt(1-ρ²)]]
    
    samples = torch.zeros(num_samples, pred_len, N, 2, device=device)
    
    # Vectorized transformation
    sqrt_term = torch.sqrt(torch.clamp(1 - corr**2, min=1e-6))
    
    # x = μx + σx * z1
    samples[:, :, :, 0] = mux.unsqueeze(0) + sx.unsqueeze(0) * z[:, :, :, 0]
    
    # y = μy + ρ*σy*z1 + σy*sqrt(1-ρ²)*z2
    samples[:, :, :, 1] = (
        muy.unsqueeze(0) + 
        corr.unsqueeze(0) * sy.unsqueeze(0) * z[:, :, :, 0] +
        sy.unsqueeze(0) * sqrt_term.unsqueeze(0) * z[:, :, :, 1]
    )
    
    return samples


def evaluate_best_of_k(V_pred, V_target, num_samples=20):
    """
    Comprehensive best-of-K evaluation (paper-aligned).
    
    Paper approach:
    1. Sample K trajectories from the predicted Gaussian distribution (once)
    2. For each sample, compute ADE (average displacement over all timesteps)
    3. Select the sample with minimum ADE (closest to ground truth)
    4. Report metrics for this SAME best sample
    
    This ensures both metrics are computed on the same trajectory, which is
    more consistent with the paper's description of selecting "the closest
    sample to the ground-truth".
    
    Args:
        V_pred: Predicted Gaussian parameters [pred_len, N, 5]
        V_target: Ground truth positions [pred_len, N, 2] (absolute coordinates)
        num_samples: Number of samples (K=20 in paper)
    
    Returns:
        dict with:
            'minADE': Minimum ADE across K samples (used for sample selection)
            'FDE': FDE of the best sample (selected by minADE criterion)
            'best_sample': The selected trajectory [pred_len, N, 2]
            'all_ade_values': ADE values for all K samples (for analysis)
    """
    # Sample K trajectories from the Gaussian (ONCE)
    samples = sample_bivariate_gaussian(V_pred, num_samples)  # [K, pred_len, N, 2]
    
    K, pred_len, N, _ = samples.shape
    
    # Expand target for broadcasting
    target_expanded = V_target.unsqueeze(0)  # [1, pred_len, N, 2]
    
    # Compute displacement at each timestep for all samples: [K, pred_len, N]
    displacements = torch.sqrt(
        (samples[:, :, :, 0] - target_expanded[:, :, :, 0])**2 + 
        (samples[:, :, :, 1] - target_expanded[:, :, :, 1])**2
    )
    
    # Compute ADE for each sample: average over timesteps and vessels [K]
    ade_per_sample = displacements.mean(dim=[1, 2])
    
    # Select best sample based on ADE (closest to ground truth)
    min_ade_idx = torch.argmin(ade_per_sample)
    best_sample = samples[min_ade_idx]  # [pred_len, N, 2]
    
    # Compute ADE for best sample
    min_ade = ade_per_sample[min_ade_idx].item()
    
    # Compute FDE for the SAME best sample
    # FDE = average displacement at final timestep only
    final_displacement = displacements[min_ade_idx, -1, :]  # [N]
    fde_best_sample = final_displacement.mean().item()
    
    return {
        'minADE': min_ade,                                      # Min ADE (used for selection)
        'FDE': fde_best_sample,                                 # FDE of best sample
        'best_sample': best_sample.detach(),                    # Selected trajectory
        'all_ade_values': ade_per_sample.detach().cpu().numpy()  # All ADE values
    }


def best_of_k_ade(V_pred, V_target, num_samples=20):
    """
    Compute minimum Average Displacement Error (minADE) with best-of-K sampling.
    
    DEPRECATED: Use evaluate_best_of_k() instead for paper-aligned evaluation.
    This function is kept for backward compatibility.
    
    Paper approach:
    1. Sample K trajectories from the predicted Gaussian distribution
    2. For each sample, compute ADE with ground truth
    3. Select the sample with minimum ADE (closest to ground truth)
    4. Return this minimum ADE as the final metric
    
    Args:
        V_pred: Predicted Gaussian parameters [pred_len, N, 5]
        V_target: Ground truth positions [pred_len, N, 2]
        num_samples: Number of samples (K=20 in paper)
    
    Returns:
        minADE: Minimum average displacement error across all samples
        best_sample: The sample trajectory with minimum error
    """
    results = evaluate_best_of_k(V_pred, V_target, num_samples)
    return results['minADE'], results['best_sample']


def best_of_k_fde(V_pred, V_target, num_samples=20):
    """
    Compute FDE of best sample selected by ADE criterion.
    
    DEPRECATED: Use evaluate_best_of_k() instead for paper-aligned evaluation.
    This function is kept for backward compatibility.
    
    Note: This returns FDE of the sample with minimum ADE, not minimum FDE.
    
    Args:
        V_pred: Predicted Gaussian parameters [pred_len, N, 5]
        V_target: Ground truth positions [pred_len, N, 2]
        num_samples: Number of samples (K=20 in paper)
    
    Returns:
        FDE of best sample (selected by minADE)
        best_sample: The sample trajectory with minimum ADE
    """
    results = evaluate_best_of_k(V_pred, V_target, num_samples)
    return results['FDE'], results['best_sample']