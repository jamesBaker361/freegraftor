import torch
import torch.nn as nn
from einops import rearrange

def get_match(src_features, tar_features, h_src=64, w_src=64, h_tar=64, w_tar=64, cyc_threshold=1.5, tar_mask=None, sim_threshold=0.2):
    src2tar_1d, sim_mask_1d = match_feature_src2tar(src_features, tar_features, tar_mask=tar_mask, sim_threshold=sim_threshold)
    src2tar_flow, src2src_2d_xy = mapping_to_flow(src2tar_1d, h_src, w_src, h_tar, w_tar)
    src2tar_flow = src2tar_flow.to(src_features)
    src2src_2d_xy = src2src_2d_xy.to(src_features)
    src2tar_2d_xy = src2src_2d_xy + src2tar_flow
    
    if cyc_threshold > 0:
        tar2src_1d, _ = match_feature_src2tar(tar_features, src_features)
        tar2src_flow, tar2tar_2d_xy = mapping_to_flow(tar2src_1d, h_tar, w_tar, h_src, w_src)
        tar2src_flow = tar2src_flow.to(src_features)
        src_tar2src_flow = nn.functional.grid_sample(rearrange(tar2src_flow, 'b h w c -> b c h w'), src2tar_2d_xy)
        flow_bias = rearrange(src2tar_flow, 'b h w c -> b c h w') + src_tar2src_flow
        bias_distance = torch.norm(flow_bias, dim=1, p=2)
        bias_mask_2d = (bias_distance < cyc_threshold).to(src_features)
        bias_mask_1d = rearrange(bias_mask_2d, 'b h w -> b (h w)', h=h_src, w=w_src)
    else:
        bias_mask_1d = None
        
    mask_1d = bias_mask_1d * sim_mask_1d
    
    return src2tar_2d_xy, mask_1d, src2tar_1d

def apply_match(src2tar_2d_xy, tar_features, h_tar=64, w_tar=64):
    tar_features_ = rearrange(tar_features, 'b (h w) c -> b c h w', h=h_tar, w=w_tar)
    new_features_ = nn.functional.grid_sample(tar_features_, src2tar_2d_xy)
    new_features = rearrange(new_features_, 'b c h w -> b (h w) c')
    return new_features

def mapping_to_flow(src2tar_1d, h_src=64, w_src=64, h_tar=64, w_tar=64):
    src2tar_2d = rearrange(src2tar_1d, 'b (hs ws) -> b hs ws', hs=h_src, ws=w_src)
    
    src2tar_2d_y = src2tar_2d // w_tar
    src2tar_2d_x = src2tar_2d % w_tar
    src2tar_2d_y_normed = (src2tar_2d_y / h_tar) * 2 - 1 + 1 / h_tar
    src2tar_2d_x_normed = (src2tar_2d_x / w_tar) * 2 - 1 + 1 / w_tar
    src2tar_2d_xy = torch.stack([src2tar_2d_x_normed, src2tar_2d_y_normed], dim=-1)
    
    src2src_2d_y = torch.arange(0, h_src).unsqueeze(dim=-1).repeat(1, w_src).unsqueeze(dim=0)
    src2src_2d_x = torch.arange(0, w_src).unsqueeze(dim=0).repeat(h_src, 1).unsqueeze(dim=0)
    src2src_2d_y_normed = (src2src_2d_y / h_src) * 2 - 1 + 1 / h_src
    src2src_2d_x_normed = (src2src_2d_x / w_src) * 2 - 1 + 1 / w_src
    src2src_2d_xy = torch.stack([src2src_2d_x_normed, src2src_2d_y_normed], dim=-1).to(src2tar_2d_xy)
    
    src2tar_flow = src2tar_2d_xy - src2src_2d_xy
    smoothed_src2tar_flow = smooth_flow(src2tar_flow)
    return smoothed_src2tar_flow, src2src_2d_xy

def smooth_flow(flow):
    return flow

def match_feature_src2tar(src_features, tar_features, tar_mask=None, sim_threshold=0.5):
    # Find most similar pixel in target for each source pixel
    src_features_scale = torch.norm(src_features, p=2, dim=-1).unsqueeze(dim=-1)
    tar_features_scale = torch.norm(tar_features, p=2, dim=-1).unsqueeze(dim=-2)
    src_tar_prod = torch.einsum('b m d, b n d -> b m n', src_features, tar_features)
    features_scale = src_features_scale * tar_features_scale
    sim_src2tar = src_tar_prod / (features_scale + 1e-10)
    
    if tar_mask is not None:
        tar_mask_ = (1 - tar_mask.unsqueeze(dim=0).unsqueeze(dim=0)) * (-1e10)
        sim_src2tar = sim_src2tar + tar_mask_
    
    max_sim, max_ids = torch.max(sim_src2tar, dim=2)
    sim_mask = (max_sim > sim_threshold).to(src_features)
    return max_ids, sim_mask