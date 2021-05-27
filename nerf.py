import torch
import torch.nn.functional as F

def sample_cam_poses(std_yaw, std_pitch, num:int) -> torch.Tensor:
    'return list of 4x4 matrices'
    cam2world = []
    for i in range(num):
        yaw = torch.randn(1) * std_yaw
        pitch = torch.randn(1) * std_pitch
        r = 1.
        cam2world.append(torch.tensor([ 
            [torch.cos(yaw),    -torch.sin(pitch) * torch.sin(yaw), torch.cos(pitch) * torch.sin(yaw),  r * torch.cos(pitch) * torch.sin(yaw)], 
            [0.,                torch.cos(pitch),                   torch.sin(pitch),                   r * torch.sin(pitch)],
            [-torch.sin(yaw),   -torch.sin(pitch) * torch.cos(yaw), torch.cos(pitch) * torch.cos(yaw),  r *torch.cos(pitch) * torch.cos(yaw)],
            [0.,                0.,                                 0.,                                 1.]]))
    return torch.stack(cam2world)

def get_cam_poses(yaws, pitchs, rs) -> torch.Tensor:
    'return list of 4x4 matrices'
    cam2world = []
    for yaw, pitch, r in zip(list(yaws), list(pitchs), list(rs)):
        cam2world.append(torch.tensor([ 
            [torch.cos(yaw),    -torch.sin(pitch) * torch.sin(yaw), torch.cos(pitch) * torch.sin(yaw),  r * torch.cos(pitch) * torch.sin(yaw)], 
            [0.,                torch.cos(pitch),                   torch.sin(pitch),                   r * torch.sin(pitch)],
            [-torch.sin(yaw),   -torch.sin(pitch) * torch.cos(yaw), torch.cos(pitch) * torch.cos(yaw),  r *torch.cos(pitch) * torch.cos(yaw)],
            [0.,                0.,                                 0.,                                 1.]]))
    return torch.stack(cam2world)

def get_ray_bundle(height:int, width:int, fov_y:float, cam2world:torch.Tensor):
    'return shape (height, width, 3), (height, width, 3)'
    ii, jj = torch.meshgrid(torch.arange(height).to(cam2world), torch.arange(width).to(cam2world))

    directions = torch.stack([(jj - width * 0.5) / height * 2. * fov_y, -(ii - height * 0.5) / height * 2. * fov_y, -torch.ones_like(ii)], dim=-1)
    ray_directions = torch.sum(directions[..., None, :] * cam2world[:3, :3], dim=-1)
    ray_origins = cam2world[:3, -1].expand(ray_directions.shape)
    return ray_origins, ray_directions

def compute_query_points_from_rays(
    ray_origins:torch.Tensor,
    ray_directions:torch.Tensor,
    near_thresh:float,
    far_thresh:float,
    num_samples:int,
    randomize:bool=True,
    depth_values=None,
):
    'return shape (height, width, num_samples, 3), (height, width, num_sample)'
    if depth_values is None:
        depth_values = torch.linspace(near_thresh, far_thresh, num_samples).to(ray_origins).expand(list(ray_directions.shape[:-1]) + [num_samples])
        if randomize is True:
            noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
            depth_values = depth_values + torch.rand(noise_shape).to(ray_origins) * (far_thresh - near_thresh) / num_samples
    else:
        num_samples = depth_values.shape[-1]
    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
    query_dirs = ray_directions[..., None, :].repeat([1] * (ray_directions.dim() - 1) + [num_samples, 1])
    return query_points, query_dirs, depth_values

def sample_importance(
    depth_values:torch.Tensor,
    weights:torch.Tensor,
    num_samples:int,
    randomize = True
):
    ''' return depth sampled according to weights. shape (height, width, num_samples)'''
    weights = weights + 1e-5 
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)

    if randomize:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples]).to(weights)
    else:
        u = torch.linspace(0., 1., num_samples).to(weights).expand(list(weights.shape[:-1]) + [num_samples])

    inds = torch.searchsorted(cdf.detach(), u, right=False).clamp(0, cdf.shape[-1] - 1)
    
    rng = torch.cat([depth_values[...,1:], depth_values[..., -1:]], dim=-1) - torch.cat([depth_values[..., :1], depth_values[..., :-1]], dim=-1)
    
    new_depth_values = torch.gather(depth_values, -1, inds)
    depth_rand_rng = torch.gather(rng, -1, inds)
    new_depth_values = new_depth_values + (torch.rand_like(new_depth_values) - 0.5) * depth_rand_rng
    
    return new_depth_values

def render_volume_weight(
    radiance_field:torch.Tensor,                    # (height, width, num_samples, 4)
    ray_origins:torch.Tensor,                       # (height, width, 3)
    depth_values:torch.Tensor                       # (height, width, 
):
    sigma_a = F.relu(radiance_field[..., 3])        # (height, width, num_samples)

    one_e_10 = torch.tensor([1e3], dtype=ray_origins.dtype, device=ray_origins.device)
    dists = torch.cat([depth_values[..., 1:] - depth_values[..., :-1], one_e_10.expand(depth_values[..., :1].shape)], dim=-1)
    alpha = sigma_a * dists
    weights = (1. - torch.exp(-alpha)) * torch.exp(-(torch.cumsum(alpha, dim=-1) - alpha))

    return weights            # (height, width, num_samples)

def render_volume(
    radiance_field:torch.Tensor,                    # (height, width, num_samples, 4)
    ray_origins:torch.Tensor,                       # (height, width, 3)
    depth_values:torch.Tensor                       # (height, width, 
):
    sigma_a = F.relu(radiance_field[..., 3])        # (height, width, num_samples)
    rgb = torch.sigmoid(radiance_field[..., :3])    # (height, width, num_samples, 3)

    one_e_10 = torch.tensor([1e3], dtype=ray_origins.dtype, device=ray_origins.device)
    dists = torch.cat([depth_values[..., 1:] - depth_values[..., :-1], one_e_10.expand(depth_values[..., :1].shape)], dim=-1)
    alpha = sigma_a * dists
    weights = (1. - torch.exp(-alpha)) * torch.exp(-(torch.cumsum(alpha, dim=-1) - alpha))

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(-1)

    return rgb_map, depth_map, acc_map              # (height, width, )
