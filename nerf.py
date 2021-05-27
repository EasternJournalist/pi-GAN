from numpy.core.function_base import linspace
import torch
from torch import tensor
import torch.nn.functional as F

def sample_cam_poses(std_yaw, std_pitch, num:int) -> torch.Tensor:
    'return list of 4x4 matrices'
    cam2world = []
    for i in range(num):
        yaw = torch.randn(1) * std_yaw
        pitch = torch.randn(1) * std_pitch
        r = 1. + torch.randn(1) * 0.05
        cam2world.append(torch.tensor([ 
            [torch.cos(yaw),    -torch.sin(pitch) * torch.sin(yaw), torch.cos(pitch) * torch.sin(yaw),  r * torch.cos(pitch) * torch.sin(yaw)], 
            [0.,                torch.cos(pitch),                   torch.sin(pitch),                   r * torch.sin(pitch)],
            [-torch.sin(yaw),   -torch.sin(pitch) * torch.cos(yaw), torch.cos(pitch) * torch.cos(yaw),  r *torch.cos(pitch) * torch.cos(yaw)],
            [0.,                0.,                                 0.,                                 1.]]))
    return torch.stack(cam2world)

def get_cam_pose(yaws, pitchs, rs) ->torch.Tensor:
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
    randomize = True
):
    'return shape (height, width, num_samples, 3), (height, width, num_sample)'
    depth_values = torch.linspace(near_thresh, far_thresh, num_samples).to(ray_origins).expand(list(ray_directions.shape[:-1]) + [num_samples])
    if randomize is True:
        noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
        depth_values = depth_values + torch.rand(noise_shape).to(ray_origins) * (far_thresh - near_thresh) / num_samples
    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
    query_dirs = ray_directions[..., None, :].repeat([1] * (ray_directions.dim() - 1) + [num_samples, 1])
    return query_points, query_dirs, depth_values

def sample_importance(
    ray_origins:torch.Tensor,
    ray_directions:torch.Tensor,
    depth:torch.Tensor,
    weight:torch.Tensor,
    num_samples:int,
    randomize = True
):
    weight = weight + 1e-5 # prevent nans
    pdf = weight / torch.sum(weight, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))
    
    if randomize:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.linspace(0., 1., num_samples)

        noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
        depth_values = depth_values + torch.rand(noise_shape).to(ray_origins) * (far_thresh - near_thresh) / num_samples
    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
    query_dirs = ray_directions[..., None, :].repeat([1] * (ray_directions.dim() - 1) + [num_samples, 1])
    return query_points, query_dirs, depth_values

def render_volume_density(
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


def get_image_from_nerf_model(
    model:torch.nn.Module,
    latents:torch.Tensor,
    cam2world:torch.Tensor,
    height:int,
    width:int,
    fov_y = 0.2,
    near_thresh = 0.5,
    far_thresh = 1.5,
    depth_samples_per_ray = 32
):
    assert latents.shape[0] == len(cam2world)

    images = []
    for latent, c2w in zip(latents.unbind(dim=0), cam2world.unbind(0)):   # for each (latent, c2w), generate an image
        c2w = c2w.to(latent)
        ray_origins, ray_directions = get_ray_bundle(height, width, fov_y, c2w)
        query_points, query_dirs, depth_values = compute_query_points_from_rays(ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray)

        flattened_query_points = query_points.reshape((-1, 3))
        flattened_query_dirs = query_dirs.reshape((-1, 3))
        
        assert flattened_query_points.shape[0] == flattened_query_dirs.shape[0]
        radiance_field_flattened = model(latent, flattened_query_points, flattened_query_dirs)

        unflattened_shape = list(query_points.shape[:-1]) + [4]
        radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)

        rgb_predicted, _, _ = render_volume_density(radiance_field, ray_origins, depth_values)
        image = rgb_predicted.permute((2, 0, 1))    # (h, w, c) -> (c, h, w)
        images.append(image)

    return torch.stack(images)