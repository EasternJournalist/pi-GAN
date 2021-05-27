import math

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

import torchvision
from torchvision.utils import save_image

from nerf import *
import imageio
import numpy as np

# losses
def gradient_penalty(images, output) -> torch.Tensor:
    batch_size, device = images.shape[0], images.device
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    l2 = ((gradients.norm(2, dim = 1) - 1) ** 2).mean()
    return l2

# FiLM SIREN y = sin(gamma * (w x + b) + beta)
class FiLMSIREN(nn.Module):
    def __init__(self, in_features:int, out_features:int, omega_0:float=30., is_first:bool=False, bias:bool=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            b = math.sqrt(6. / self.in_features) if self.is_first else math.sqrt(6. / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-b, b)

    def forward(self, x, gamma=None, beta=None):
        out =  self.linear(x)
        # FiLM modulation
        if gamma is not None:
            out = out * gamma
        if beta is not None:
            out = out + beta

        out = torch.sin(self.omega_0 * out)
        return out

# Equalized Learning rate initialzation for linear (in PG-GAN). Default for nn.Linear is Kaming initialization.
class EqualizedLinear(nn.Linear):
    def __init__(self, in_features:int, out_features:int, bias:bool=True, nonlinearity:str='leaky_relu', *args, **kwargs):
        self.nonlinearity = nonlinearity
        super(EqualizedLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias, *args, **kwargs)

    def reset_parameters(self):
        std = nn.init.calculate_gain(self.nonlinearity) / math.sqrt(self.in_features)
        nn.init.normal_(self.weight, mean=0., std=std)
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)


# Mapping network
class MappingNetwork(nn.Module):    
    def __init__(self, dim_in:int=128, dim_hidden:int=256, num_hidden:int=3, dim_out=[256 for i in range(8)]+[128]):
        super(MappingNetwork, self).__init__()

        mlp_layers = [EqualizedLinear(dim_in, dim_hidden), nn.LeakyReLU(negative_slope=0.2)]
        for i in range(num_hidden - 1):
            mlp_layers.append(EqualizedLinear(dim_hidden, dim_hidden))
            mlp_layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.mlp = nn.Sequential(*mlp_layers)
        
        self.to_gammas = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(dim_hidden, d)
            )
             for d in dim_out
        ])
        self.to_betas = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(dim_hidden, d)
            )
             for d in dim_out
        ])
        self.dim_out = dim_out

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        x = self.mlp(x)

        gammas = [l(x) * 2. for l in self.to_gammas] + [None] * (len(self.dim_out) - len(self.to_gammas))
        betas = [l(x) * 0.1 for i, l in enumerate(self.to_betas)] + [None] * (len(self.dim_out) - len(self.to_betas))
        
        return gammas, betas


# generator
class SIRENGenerator(nn.Module):
    def __init__(self, *, dim_latent:int=128, dim_hidden:int=256, siren_num_layers:int=8):
        super().__init__()
        dim_x = 3
        dim_d = 3

        self.mapping = MappingNetwork(dim_in=dim_latent, dim_hidden=dim_hidden, dim_out=[dim_hidden] * siren_num_layers)

        self.filmsiren_series = nn.ModuleList(
            [FiLMSIREN(in_features=dim_x, out_features=dim_hidden, is_first=True)] + [FiLMSIREN(in_features=dim_hidden, out_features=dim_hidden) for i in range(siren_num_layers - 1)]
        )
        self.to_alpha = nn.Linear(dim_hidden, 1)

        self.to_rgb_siren = FiLMSIREN(in_features=dim_hidden + dim_d, out_features=dim_hidden // 2)
        self.to_rgb_linear = nn.Linear(dim_hidden // 2, 3)
        

    def forward(self, latent_z, coords_x, views_d, batch_size:int=8192):
        latent_z = latent_z.view(1, -1)

        gammas, betas = self.mapping(latent_z)
          
        outs = []
        for x, d in zip(coords_x.split(batch_size), views_d.split(batch_size)):
            for i, l in enumerate(self.filmsiren_series):
                x = l(x, gammas[i], betas[i])
            alpha = self.to_alpha(x)
            
            x = torch.cat([x, d], dim=1)
            x = self.to_rgb_siren(x, None, None)
            rgb = self.to_rgb_linear(x)

            out = torch.cat((rgb, alpha), dim=-1)
            outs.append(out)
            
        return torch.cat(outs)
        

class ImageGenerator(nn.Module):
    def __init__(
        self,
        image_size,
        dim_latent,
        dim_hidden,
        siren_num_layers,
        device_ids
    ):
        super().__init__()
        self.dim_latent = dim_latent
        self.image_size = image_size
        self.device_ids = device_ids
        
        self.nerf_model = SIRENGenerator(
            dim_latent=dim_latent,
            dim_hidden=dim_hidden,
            siren_num_layers=siren_num_layers
        )

    def set_image_size(self, image_size):
        self.image_size = image_size

    def forward(self, latents:torch.Tensor, camera_poses:torch.Tensor, with_importance=True):
        image_size = self.image_size

        generated_images = self.get_image_from_nerf_model(
            latents,
            camera_poses,
            image_size,
            image_size,
            with_importance=with_importance
        )

        return generated_images
    
    def get_image_from_nerf_model(
        self,
        latents:torch.Tensor,
        cam2world:torch.Tensor,
        height:int,
        width:int,
        fov_y=0.2,
        near_thresh:float=0.7,
        far_thresh:float=1.3,
        samples_per_ray=24,
        with_importance:bool=True
    ):
        assert latents.shape[0] == cam2world.shape[0]

        images = []
        depth_images = []
        
        for latent, c2w in zip(latents.unbind(dim=0), cam2world.unbind(0)):   # for each (latent, c2w), generate an image
            c2w = c2w.cuda() 
            ray_origins, ray_directions = get_ray_bundle(height, width, fov_y, c2w)

            if with_importance:
                # uniform sample 1/2
                query_points, query_dirs, depth_values = compute_query_points_from_rays(ray_origins, ray_directions, near_thresh, far_thresh, samples_per_ray // 2, False, None)
                flattened_query_points = query_points.reshape((-1, 3))
                flattened_query_dirs = query_dirs.reshape((-1, 3))
                
                with torch.no_grad():
                    radiance_field_flattened = self.nerf_model(latent, flattened_query_points, flattened_query_dirs).detach()
                unflattened_shape = list(query_points.shape[:-1]) + [4]
                radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)
                weights = render_volume_weight(radiance_field, ray_origins, depth_values)

                # importance sample 1/2
                new_depth_values = sample_importance(depth_values, weights,  samples_per_ray // 2, randomize=True)
                depth_values = torch.cat([depth_values, new_depth_values], dim=-1).sort()[0]
                query_points, query_dirs, new_depth_values = compute_query_points_from_rays(ray_origins, ray_directions, near_thresh, far_thresh, samples_per_ray, False, depth_values)
            else:
                query_points, query_dirs, depth_values = compute_query_points_from_rays(ray_origins, ray_directions, near_thresh, far_thresh, samples_per_ray, True, None)
            
            flattened_query_points = query_points.reshape((-1, 3))
            flattened_query_dirs = query_dirs.reshape((-1, 3))
            
            radiance_field_flattened = self.nerf_model(latent, flattened_query_points, flattened_query_dirs)
            unflattened_shape = list(query_points.shape[:-1]) + [4]
            radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)
            rgb_predicted, depth_predicted, _ = render_volume(radiance_field, ray_origins, depth_values)
            image = rgb_predicted.permute((2, 0, 1))    # (h, w, c) -> (c, h, w)
            
            images.append(image)
            depth_images.append(depth_predicted)

        return torch.stack(images), torch.stack(depth_images)

# CoordConv
class CoordConv2D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, padding:int=1, with_r:bool=False):
        super().__init__()
        self.in_channel = in_channels
        self.with_r = with_r
        self.conv = nn.Conv2d(in_channels=in_channels + (2 if not with_r else 3), out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input_tensor:torch.Tensor):
        batch_size, _, y_dim, x_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1) * 2. - 1.
        yy_channel = yy_channel.float() / (y_dim - 1) * 2. - 1.

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        x = torch.cat([input_tensor, xx_channel.type_as(input_tensor), yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            x = torch.cat([x, rr], dim=1)

        x = self.conv(x)
        return x

# Discriminator
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.net = nn.Sequential(
            CoordConv2D(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            CoordConv2D(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.pooling = nn.AvgPool2d(2)

    def forward(self, x):
        res = self.res(x)
        x = self.net(x)
        x = self.pooling(x)
        x = x + res
        return x

class Discriminator(nn.Module):
    def __init__(self, init_resolution:int=16, final_resolution:int=64):
        super().__init__()
        assert math.log2(init_resolution).is_integer() and math.log2(final_resolution).is_integer(), 'resolution must be a power of 2'

        self.init_resolution = init_resolution
        self.final_resolution = final_resolution

        log2_init_res = int(math.log2(init_resolution))
        log2_final_res = int(math.log2(final_resolution))
        
        self.resolutions = [final_resolution // 2**i for i in range(0, log2_final_res - 1)] # Input resolution of each block
        chans = [min(64*2**i, 400) for i in range(0, log2_final_res - 1)]                   # Output channels of each block

        self.conv_blocks = nn.ModuleList(
            [DiscriminatorBlock(in_channels=(chans[i - 1] if i > 0 else chans[0]), out_channels=chans[i]) for i in range(0, len(chans))]
        )
        
        # adaper_blocks[i] before conv_blocks[i]
        self.adapter_blocks = nn.ModuleList([
            nn.Sequential(
                CoordConv2D(3, chans[i - 1] if i > 0 else chans[0]), 
                nn.LeakyReLU(negative_slope=0.2)
            ) for i in range(0, log2_final_res - log2_init_res + 1)
        ])
        
        self.final_conv = nn.Conv2d(in_channels=400, out_channels=1, kernel_size=2)
        
        self.fadein = False
        self.fadein_iters = 6000
        self.alpha = 0.
        self.iterations = 0
        self.cur_resolution = init_resolution

    def set_resolution(self, new_resolution, fadein=True):
        if new_resolution >= self.final_resolution:
            return
        self.fadein = fadein
        self.alpha = 0.
        self.iterations = 0
        self.cur_resolution = new_resolution

    def update_iter_(self):
        if not self.fadein:
            return
        self.iterations += 1
        self.alpha += (1 / self.fadein_iters)
        if self.alpha >= 1.:
            self.fadein = False
            self.alpha = 1.

    def forward(self, img:torch.Tensor):
        for i in range(len(self.conv_blocks)):
            if self.cur_resolution < self.resolutions[i]:
                continue
            if self.cur_resolution == self.resolutions[i]:
                x = self.adapter_blocks[i](img)

            if self.cur_resolution // 2 == self.resolutions[i] and self.fadein:
                x_down = F.avg_pool2d(input=img, kernel_size=2) # F.interpolate(img, scale_factor = 0.5, mode=)
                x = x * self.alpha + self.adapter_blocks[i](x_down) * (1. - self.alpha)

            x = self.conv_blocks[i](x)
        
        out = self.final_conv(x)
        return out


def cycle(iterable):
    while True:
        for i in iterable:
            yield i

            
# pi-GAN class
class piGAN:
    dataloader:DataLoader

    def __init__(self, dataset, device_ids=[0, 1, 2, 3]):
        self.image_size = 64
        self.G = nn.DataParallel(ImageGenerator(
            image_size=16,
            dim_latent=128,
            dim_hidden=256,
            siren_num_layers=8,
            device_ids=device_ids
        ), device_ids=device_ids).cuda()

        self.D = Discriminator(
            init_resolution=16,
            final_resolution=self.image_size
        )
        self.D = nn.DataParallel(self.D, device_ids=device_ids).cuda()

        self.optim_G = Adam(self.G.parameters(), betas=(0, 0.9), lr=5e-5)
        self.optim_D = Adam(self.D.parameters(), betas=(0, 0.9), lr=4e-4)
        
        self.lr_scheduler_G =lr_scheduler.ExponentialLR(self.optim_G, gamma=0.2**(1./30000))
        self.lr_scheduler_D =lr_scheduler.ExponentialLR(self.optim_D, gamma=0.25**(1./30000))

        self.loss_D = 0.
        self.loss_G = 0.
        self.loss_gp = 0.

        self.batch_size_D = 128
        self.batch_size_G = 64
        self.with_gradient_penalty = True

        self.iterations = 1
        self.grow_iters = 10000

        self.dataset = dataset 
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size_D, shuffle=True)
        
        self.cur_resolution = 16

    def train(self, num_iters):
        self.dataloader_cycle = cycle(self.dataloader)
        
        for i in range(num_iters):
            if i % 100 == 0:
                tot_loss_D = 0.
                tot_loss_G = 0.
                tot_loss_gp = 0.
            i_mod = i % 100 + 1
            
            self.train_step()
            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()
            
            tot_loss_G += self.loss_G
            tot_loss_D += self.loss_D
            tot_loss_gp += self.loss_gp
            
            print(f'[Iter {self.iterations:>6d}] G loss:{tot_loss_G / i_mod:>5f}, D loss:{tot_loss_D / i_mod:>5f}, gp loss:{tot_loss_gp / i_mod:>5f}', end='\r')
            
            if self.iterations % 100 == 0:
                print()
                
            if self.iterations % 100 == 0:
                print('[Test]')
                self.test_imgs()
                self.test_video(traj='circle')
                self.test_video(traj='straight')
                
            
            if self.iterations % 1000 == 0:
                print('[Save check point]')
                self.save_ckpt(f'/output/ckpt{str(self.iterations).zfill(6)}.pth')

            self.iterations += 1

    def train_step(self):
        if self.iterations % self.grow_iters == 0 and self.cur_resolution < self.image_size:
            print(f'Resolution Grow to {self.cur_resolution * 2}')
            self.cur_resolution *= 2
            if self.iterations != 0:
                self.D.module.set_resolution(self.cur_resolution)

            self.G.module.set_image_size(self.cur_resolution)
            self.dataset.create_transform(self.cur_resolution)
            self.batch_size_G = 16384 // (self.cur_resolution * self.cur_resolution)
            
        # Train Discriminator
        self.D.train()
        
        tiny_steps = 1
        self.loss_D = 0.
        i = 0
        while i == 0 or (i < tiny_steps and self.loss_D > self.loss_G):
            i += 1
        
            real_imgs = next(self.dataloader_cycle)
            real_imgs = real_imgs.cuda().requires_grad_()
            real_imgs_D_out = self.D(real_imgs)
            with torch.no_grad():
                rand_latents = torch.randn(self.batch_size_D, self.G.module.dim_latent).cuda()
                cam2world = sample_cam_poses(0.3, 0.15, self.batch_size_D).cuda()
                fake_imgs, _ = self.G(rand_latents, cam2world, with_importance=self.iterations>15000)
            fake_imgs.detach_()
            
            fake_imgs_D_out = self.D(fake_imgs)
            
            loss = torch.mean(F.softplus(fake_imgs_D_out)) + torch.mean(F.softplus(-real_imgs_D_out))
            
            self.loss_D += loss.item() / tiny_steps
            if self.with_gradient_penalty:
                gp = gradient_penalty(real_imgs, real_imgs_D_out)
                self.loss_gp = gp.item()
                loss = loss + 10. * gp
            
            self.optim_D.zero_grad()
            loss.backward()
            self.optim_D.step()

        # Train Generator
        tiny_steps = 1
        
        self.G.train()
        i = 0
        while i == 0 or (i < tiny_steps and self.loss_G > self.loss_D * 2.):
            i += 1
            #print(f'G [{i:5d}/{tiny_steps:5d}]', end='\r')
            rand_latents = torch.randn(self.batch_size_G // 2, self.G.module.dim_latent).cuda().repeat((2, 1))
            cam2world = sample_cam_poses(0.3, 0.15, self.batch_size_G).cuda()
            fake_imgs, _ = self.G(rand_latents, cam2world, with_importance=self.iterations>15000)
            
            loss = torch.mean(F.softplus(-self.D(fake_imgs)))
            
            self.loss_G = loss.item()
            
            self.optim_G.zero_grad()
            loss.backward()
            self.optim_G.step()
            
        self.D.module.update_iter_()

    def test_imgs(self):
        '''4 x 4 = 16 images of currenct resolution'''
        self.G.eval()
        with torch.no_grad():
            rand_latents = torch.randn(4, self.G.module.dim_latent).cuda()
            rand_latents = rand_latents[:, None, :].repeat((1, 4, 1)).reshape(16, -1)
            cam2world = get_cam_poses(torch.linspace(-0.3, 0.3, 4), torch.zeros(4), torch.ones(4)).cuda()
            cam2world = cam2world.repeat((4, 1, 1))
            fake_imgs, depth_imgs = self.G(rand_latents, cam2world)
        
        depth_imgs = torch.exp(- 3. * depth_imgs + 2.)[:, None, :, :]
        
        torchvision.utils.save_image(fake_imgs, f'/output/test/iter_{str(self.iterations).zfill(6)}.png', nrow=4)
        torchvision.utils.save_image(depth_imgs, f'/output/test/iter_{str(self.iterations).zfill(6)}_depth.png', nrow=4)
    
    def test_video(self, traj:str='circle'):
        self.G.eval()
        with torch.no_grad():
            rand_latents = torch.randn(1, self.G.module.dim_latent).cuda()
            rand_latents = rand_latents[:, :].repeat((128, 1))
            
            if traj == 'circle':
                theta = torch.linspace(0, 6.28, 128)
                cam2world = get_cam_poses(0.3 * torch.cos(theta), 0.3 * torch.sin(theta), torch.ones(128)).cuda()
            elif traj == 'straight':
                cam2world = get_cam_poses(torch.linspace(-0.6, 0.6, 128), torch.zeros(128), torch.ones(128)).cuda()
                
            fake_imgs, _ = self.G(rand_latents, cam2world)
            
        fake_imgs = (fake_imgs.permute((0, 2, 3, 1)).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        save_path = f'/output/test/iter_{str(self.iterations).zfill(6)}_{traj}.mp4'
        imageio.mimwrite(save_path, fake_imgs, fps=30, quality=8) 
    
    def save_ckpt(self, path):

        state = {
            'G':self.G.module.state_dict(), 
            'D':self.D.module.state_dict(),
            'optim_G':self.optim_G.state_dict(),
            'optim_D':self.optim_D.state_dict(),
            'lr_scheduler_G':self.lr_scheduler_G.state_dict(),
            'lr_scheduler_D':self.lr_scheduler_D.state_dict(),
            'iterations':self.iterations,
            
            'cur_resolution':self.cur_resolution,
            'fadein':self.D.module.fadein,
            'fadein_alpha':self.D.module.alpha,
            'fadein_iterations':self.D.module.iterations,
            }
        torch.save(state, path)
        

    def load_ckpt(self, path): 
        state = torch.load(path)

        self.G.module.load_state_dict(state['G'])
        self.D.module.load_state_dict(state['D'])
        self.optim_G.load_state_dict(state['optim_G'])
        self.optim_D.load_state_dict(state['optim_D'])
        self.lr_scheduler_G.load_state_dict(state['lr_scheduler_G'])
        self.lr_scheduler_D.load_state_dict(state['lr_scheduler_D'])
        self.iterations = state['iterations']
        
        self.cur_resolution = state['cur_resolution'] if 'cur_resolution' in state.keys() else 64
        self.D.module.cur_resolution = self.cur_resolution
        self.D.module.fadein = state['fadein'] if 'fadein' in state.keys() else False
        self.D.module.alpha = state['fadein_alpha'] if 'fadein_alpha' in state.keys() else 0.
        self.D.module.iterations = state['fadein_iterations'] if 'fadein_iterations' in state.keys() else 0
        
        
        self.G.module.set_image_size(self.cur_resolution)
        self.dataset.create_transform(self.cur_resolution)
        self.batch_size_G = 16384 // (self.cur_resolution * self.cur_resolution)


    def reset_optimizer(self):
        self.optim_G = Adam(self.G.parameters(), betas=(0, 0.9), lr=0.4e-5)
        self.optim_D = Adam(self.D.parameters(), betas=(0, 0.9), lr=0.4e-4)
        
        self.lr_scheduler_G =lr_scheduler.ExponentialLR(self.optim_G, gamma=0.2**(1./20000))
        self.lr_scheduler_D =lr_scheduler.ExponentialLR(self.optim_D, gamma=0.25**(1./20000))
