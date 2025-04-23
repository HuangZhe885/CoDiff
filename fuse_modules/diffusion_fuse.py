# -*- coding: utf-8 -*-
# Author: Zhe Huang <2021000892@ruc.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
import torch as th
from torch import nn, einsum, sqrt
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import  rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

import numpy as np

from PIL import Image
from tqdm.auto import tqdm

from functools import partial, wraps

from torch.special import expm1

from ema_pytorch import EMA
from accelerate import Accelerator

from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.models.fuse_modules.fuse_utils import regroup as Regroup
from opencood.models.fuse_modules.fuse_utils import regroup_with_padding as Regroup_with_padding


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules
# 
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[-1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3, # 输出通道数, RGB 为 3, 特征图的话需要调整
        self_condition = False, # 是否自我条件，后续需要改为给定的条件(即特征图)
        given_condition = True, # 是否给定条件
        learned_variance = False, # 是否学习方差
        learned_sinusoidal_cond = False, # 是否学习正弦条件
        random_fourier_features = False,    # 是否随机正弦特征
        learned_sinusoidal_dim = 16 # 学习正弦维度
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        self.given_condition = given_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3) # 初始卷积层 (init_conv) 使用 7 的核大小和 3 的填充以保持空间维度。

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)] # 维度列表 [init_dim, dim * dim_mults[0], dim * dim_mults[1], ...]
        in_out = list(zip(dims[:-1], dims[1:])) # 输入输出维度对 [(init_dim, dim * dim_mults[0]), (dim * dim_mults[0], dim * dim_mults[1]), ...]

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([  # 下采样 [ResnetBlock, ResnetBlock, Attention, Downsample] 或 [ResnetBlock, ResnetBlock, Attention, nn.Conv2d]
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),  
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]  # 中间块维度, 包含一个 ResNet 块、一个注意层和另一个 ResNet 块
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        # 上采样 [ResnetBlock, ResnetBlock, Attention, Upsample] 或 [ResnetBlock, ResnetBlock, Attention, nn.Conv2d]
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim = time_dim) # 最终残差块 将最后的上采样块与初始特征结合
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)  # 最终的 ResNet 块和卷积层生成最终输出

        self.padding = nn.ZeroPad2d((0, 1, 0, 1))  # Pad right and bottom if necessary
        

    def forward(self, x, time, x_self_cond = None, condition = None, given_cond = False):

        # print("x",x.shape)   
        # print("condition",condition.shape)
        #x torch.Size([1, 8, 104, 352])
        #condition torch.Size([1, 8, 104, 352])
        if given_cond:  
            x = torch.cat((condition, x), dim = 1)

        x = self.init_conv(x)  # 初始卷积层
        r = x.clone()  # 保存初始特征, 用于最终残差块

        t = self.time_mlp(time)  # 时间嵌入

        h = []

        pad_list = []
        # pad_h = 0
        # pad_w = 0
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)
            # print("x",x.shape)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
    

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# diffusion helpers

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# logsnr schedules and shifting / interpolating decorators
# only cosine for now

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def logsnr_schedule_cosine(t, logsnr_min = -15, logsnr_max = 15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))

def logsnr_schedule_shifted(fn, image_d, noise_d):
    shift = 2 * math.log(noise_d / image_d)
    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal shift
        return fn(*args, **kwargs) + shift
    return inner

def logsnr_schedule_interpolated(fn, image_d, noise_d_low, noise_d_high):
    logsnr_low_fn = logsnr_schedule_shifted(fn, image_d, noise_d_low)
    logsnr_high_fn = logsnr_schedule_shifted(fn, image_d, noise_d_high)

    @wraps(fn)
    def inner(t, *args, **kwargs):
        nonlocal logsnr_low_fn
        nonlocal logsnr_high_fn
        return t * logsnr_low_fn(t, *args, **kwargs) + (1 - t) * logsnr_high_fn(t, *args, **kwargs)

    return inner

# main gaussian diffusion class

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: Unet,
        *,
        image_size = (64, 128),
        channels = 256,  #
        pred_objective = 'eps',   # 
        noise_schedule = logsnr_schedule_cosine,
        noise_d = None,
        noise_d_low = None,
        noise_d_high = None,
        num_sample_steps = 500, # 
        clip_sample_denoised = True,
        min_snr_loss_weight = True,
        min_snr_gamma = 5
    ):
        super().__init__()
        assert pred_objective in {'v', 'eps'}, 'whether to predict v-space (progressive distillation paper) or noise'

        self.model = model

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # training objective

        self.pred_objective = pred_objective

        # noise schedule

        assert not all([*map(exists, (noise_d, noise_d_low, noise_d_high))]), 'you must either set noise_d for shifted schedule, or noise_d_low and noise_d_high for shifted and interpolated schedule'

        # determine shifting or interpolated schedules

        self.log_snr = noise_schedule

        if exists(noise_d):
            self.log_snr = logsnr_schedule_shifted(self.log_snr, image_size, noise_d)

        if exists(noise_d_low) or exists(noise_d_high):
            assert exists(noise_d_low) and exists(noise_d_high), 'both noise_d_low and noise_d_high must be set'

            self.log_snr = logsnr_schedule_interpolated(self.log_snr, image_size, noise_d_low, noise_d_high)

        # sampling

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

        # loss weight

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma


        # 下面都是ddim的
        '''
        betas = get_named_beta_schedule("cosine", num_sample_steps)
        self.betas = np.array(betas, dtype=np.float64)

        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        '''


    @property
    def device(self):
        return next(self.model.parameters()).device

    # 根据当前时间步和下一个时间步计算模型的均值和方差
    def p_mean_variance(self, x, time, time_next):

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])
        pred = self.model(x, batch_log_snr) 

        if self.pred_objective == 'v':
            x_start = alpha * x - sigma * pred

        elif self.pred_objective == 'eps':
            x_start = (x - sigma * pred) / alpha

        x_start.clamp_(-1., 1.)

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance
    
    # 带有条件的一次采样
    def p_mean_variance_condition(self, x, time, time_next, condition):

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])
        # condition #[1, 8, 100, 352]
        pred = self.model(x, batch_log_snr, condition = condition, given_cond = True)  # todo:: 加入条件

        if self.pred_objective == 'v':
            x_start = alpha * x - sigma * pred

        elif self.pred_objective == 'eps':
            x_start = (x - sigma * pred) / alpha

        x_start.clamp_(-1., 1.)

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    # sampling related functions
    # 根据模型的均值和方差从噪声中采样
    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x = x, time = time, time_next = time_next)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise
    

    def p_sample_condition(self, x, time, time_next, condition):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance_condition(x = x, time = time, time_next = time_next, condition = condition)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    # 采样循环
    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch = shape[0]

        img = torch.randn(shape, device = self.device)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device = self.device)

        for i in tqdm(range(self.num_sample_steps), desc = 'sampling loop time step', total = self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)

        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img
    
    def p_sample_loop_condition(self, shape, condition=None): # condition [B, L+1, C, H, W]
        batch = shape[0]
       
        B, L, C, H, W = condition.shape


        # img = torch.randn(shape, device = self.device)
        # 取condition中[B,0,C,H,W]
        img = condition[:, 0].reshape(batch, *condition.shape[2:])
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device = self.device)

        # Calculate step intervals for changing conditions
        step_interval = self.num_sample_steps // L
        change_steps = [i * step_interval for i in range(L)]

        # Sampling loop
        current_condition = condition[:, 0].reshape(batch, *condition.shape[2:])

        for i in range(self.num_sample_steps):
        # for i in range(self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]

            if i in change_steps:
                condition_index = change_steps.index(i)
                current_condition = condition[:, condition_index].reshape(batch, *condition.shape[2:])

            img = self.p_sample_condition(img, times, times_next, current_condition)

        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        # return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))
        return self.p_sample_loop((batch_size, self.channels, self.image_size[0], self.image_size[1]))
    
    @torch.no_grad()
    def sample_condition(self, batch_size = 16, condition = None):
        # return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))
        return self.p_sample_loop_condition((batch_size, self.channels, self.image_size[0], self.image_size[1]), condition = condition)
        # return self.p_sample_loop_condition((batch_size, self.channels, self.image_size[0], self.image_size[1]))

    
    @torch.no_grad()
    def ddim_sample_condition(self, batch_size = 16, condition = None):
        # return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))
        return self.ddim_sample_loop(shape=(batch_size, self.channels, self.image_size[0], self.image_size[1]), condition = condition)
        # return self.p_sample_loop_condition((batch_size, self.channels, self.image_size[0], self.image_size[1]))
    
    # training related functions - noise prediction
    #  对初始图像进行加噪，生成中间噪声图像
    @autocast(enabled = False)
    def q_sample(self, x_start, times, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma

        return x_noised, log_snr
    

    ##  add ddim
    '''
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
            return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t- pred_xstart
            ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        #函数_predict_eps_from_xstart用于反推出epsilon
        #pred_xstart是预测的起始点
    ## ws add ddim
    def ddim_sample(
        self,
        x,
        t,
        t_next,
        condition,
        eta=0.0,
    ):
        """
        使用DDIM从模型中对 x_{t-1} 进行采样。

        与 p_sample() 的用法相同。
        """
        # 使用 p_mean_variance 函数计算均值和方差
        #得到前一时刻的均值和方差
        out = self.p_mean_variance_condition(x = x, time = t, time_next = t_next, condition = condition)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
            
        # _predict_eps_from_xstart用于根据x_t,t和反推出epsilon
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
    
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)

        # 计算用于注入噪声的 sigma
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        #利用DDIM论文的公式12进行计算
        noise = th.randn_like(x)
        # 计算均值预测
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
        #t不等于0的时候需要有噪音，t等于0即最后一个时刻时直接输出均值
        #最后一个时刻直接预测的是期望值
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        # 生成样本，即重采样得到的结果
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    ## hz add ddim
    def ddim_reverse_sample(
        self,
        x,
        t,
        t_next,
        condition,
        eta=0.0,
    ):
        """
        使用DDIM反向ODE从模型中对 x_{t+1} 进行采样。
        """
        # 确保 eta 为 0 用于反向ODE
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        # 使用 p_mean_variance 函数计算均值和方差
        out = self.p_mean_variance_condition(x = x, time = t, time_next = t_next, condition = condition)
        # 计算用于采样的 epsilon
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # 计算均值预测
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}
    
    ## hz add ddim
    def ddim_sample_loop(
        self,
        shape,
        condition,
        noise=None,
        progress=False,
        eta=0.0,
    ):
        """
        从模型中使用DDIM生成样本。

        与 p_sample_loop() 的用法相同。
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            shape,
            condition,
            noise=noise,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    ## ws add ddim
    def ddim_sample_loop_progressive(
        # 一个完整的DDIM采样，从高斯白噪声到目标样本（循环过程，采样）
        self,
        shape,
        condition,
        noise=None,
        progress=False,
        eta=0.0,
    ):
        """
        使用DDIM从模型中采样, 并在每个DDIM时间步骤中产生中间样本。

        与 p_sample_loop_progressive() 的用法相同。
        """
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device = self.device)
        indices = list(range(self.num_timesteps))[::-1]
        # num_timesteps长度实际就是beta的长度
        # 若有respace则num_timesteps是一个新的时间序列长度

        if progress:
            # 惰性导入以避免依赖tqdm。
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device = self.device)
            t_next = th.tensor([i+1] * shape[0], device = self.device)
            with th.no_grad():
                out = self.ddim_sample(
                    img,
                    t,
                    t_next,
                    condition,
                    eta=eta,
                )
                yield out
                img = out["sample"]
    '''


    
    # 计算损失，用于训练过程
    def p_losses(self, x_start, times, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x, log_snr = self.q_sample(x_start = x_start, times = times, noise = noise)
        # model_out = self.model(x, log_snr)
        model_out = self.model(x, log_snr, condition = x, given_cond = True)  # todo:: 加入条件

        if self.pred_objective == 'v':
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
            target = alpha * noise - sigma * x_start

        elif self.pred_objective == 'eps':
            target = noise

        loss = F.mse_loss(model_out, target, reduction = 'none')

        loss = reduce(loss, 'b ... -> b', 'mean')

        snr = log_snr.exp()

        maybe_clip_snr = snr.clone()
        if self.min_snr_loss_weight:
            maybe_clip_snr.clamp_(max = self.min_snr_gamma)

        if self.pred_objective == 'v':
            loss_weight = maybe_clip_snr / (snr + 1)

        elif self.pred_objective == 'eps':
            loss_weight = maybe_clip_snr / snr

        return (loss * loss_weight).mean()

    def p_losses_with_condition(self, x_start, times, condition = None,noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x, log_snr = self.q_sample(x_start = x_start, times = times, noise = noise)
        # model_out = self.model(x, log_snr)
        model_out = self.model(x, log_snr, condition = condition, given_cond = True)  # todo:: 加入条件

        if self.pred_objective == 'v':
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
            target = alpha * noise - sigma * x_start

        elif self.pred_objective == 'eps':
            target = noise

        loss = F.mse_loss(model_out[:,:,2:102,2:254], target[:,:,2:102,2:254], reduction = 'none')

        loss = reduce(loss, 'b ... -> b', 'mean')

        # print(loss.item())

        snr = log_snr.exp()

        maybe_clip_snr = snr.clone()
        if self.min_snr_loss_weight:
            maybe_clip_snr.clamp_(max = self.min_snr_gamma)

        if self.pred_objective == 'v':
            loss_weight = maybe_clip_snr / (snr + 1)

        elif self.pred_objective == 'eps':
            loss_weight = maybe_clip_snr / snr

        return (loss * loss_weight).mean()

    def forward(self, img, condition, *args, **kwargs):
        # b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        img = normalize_to_neg_one_to_one(img) # 归一化到-1到1之间
        times = torch.zeros((img.shape[0],), device = self.device).float().uniform_(0, 1)

        B, L = condition.shape[:2]

        # 当前采样得到的步数,并转化为numpy
        steps = self.num_sample_steps * times
        step_interval = self.num_sample_steps // L
        change_steps = [i * step_interval for i in range(L)]  # 每次改变条件的步数


        # 计算每个batch的条件
        current_condition = condition[:, 0].reshape(B, *condition.shape[2:]) # shape [B, 8, 104, 256]

        
        for batch_index in range(B):
            for i_index, i in enumerate(change_steps):
                if steps[batch_index] >= change_steps[i_index]:
                    current_condition[batch_index] = condition[batch_index, i_index]
                    break

        return self.p_losses_with_condition(img, times, current_condition, *args, **kwargs)
    


class DifussionFusion(nn.Module):
    def __init__(self, args):
        super(DifussionFusion, self).__init__()

        self.channels = 8

        if 'train_mode' in args.keys():
            self.train_mode = args['train_mode']
        else:
            self.train_mode = 'finetune'
        
        if 'diffusion_train_steps' in args.keys():
            if self.train_mode == 'finetune':
                self.steps = args['diffusion_test_steps']
            else:
                self.steps = args['diffusion_train_steps']
        else:
            self.steps = 500

        self.unet = Unet(
            dim = 104, 
            channels=self.channels, 
            self_condition=True) # True
        
        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size= (104, 256), #dair
            # image_size= (104, 352), # v2xSet,opv2v
            channels=self.channels,
            num_sample_steps=self.steps) # 500 when training
        
        # 一个卷积层, 将[8, 8, 200, 504]转化为[8, 8, 100, 252]
        self.conv_layer_downsample = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def sample(self, x, heter_feature_2d, record_len, affine_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor  fused_feature
            input data, (B, C, H, W)
        
        heter_feature_2d : torch.Tensor
        input data, (sum(n_cav), C, 2*H, 2*W)
            
        record_len : list
            shape: (B)
            
        affine_matrix : torch.Tensor
            normalized affine matrix from 'normalize_pairwise_tfm'
            shape: (B, L, L, 2, 3) 
            
        """
        # feature map 压缩

        # heter_feature_2d =  2, 8, 200, 704]
        heter_feature_2d = self.relu(self.conv_layer_downsample(heter_feature_2d))
        # heter_feature_2d =    [2, 8, 100, 352] 


        # 1. Regroup heter_feature_2d
        _, C, H, W = heter_feature_2d.shape
        B, L = affine_matrix.shape[:2]       # 这里L是最大数量6，具体的邻居车辆需要根据record_len确定，无需额外考虑

        regroup_feature, mask = Regroup(heter_feature_2d, record_len, L)
        regroup_feature_new = [] 
        for b in range(B):
            ego = 0
            regroup_feature_new.append(warp_affine_simple(regroup_feature[b], affine_matrix[b, ego], (H, W)))
        regroup_feature = torch.stack(regroup_feature_new) # [B, L, C, H, W]


        regroup_feature = torch.cat([x.unsqueeze(1), regroup_feature], dim=1)

        
        # 2. Diffusion
        batch_size = regroup_feature.shape[0]
        cav_num = regroup_feature.shape[1]
        

        regroup_feature = F.pad(regroup_feature, (2, 2, 2, 2), "constant", 0)  #dair
        # regroup_feature = F.pad(regroup_feature, (0, 0, 2, 2), "constant", 0) # opv2v, v2xset
        

        sampled_feature = self.diffusion.sample_condition(
            batch_size=batch_size,
            condition=regroup_feature  # torch.Size([1, 6, 8, 100, 352]
        )

        # #ws DDIM
        # sampled_feature = self.diffusion.ddim_sample_condition(
        #     batch_size=batch_size,
        #     condition=regroup_feature  # torch.Size([1, 6, 8, 100, 352]
        # )

       

        sampled_feature = sampled_feature[:,:,2:102,2:254] #dair
        # sampled_feature = sampled_feature[:,:,2:102,:] # opv2v, v2xset

        return sampled_feature
    
    def forward(self, x, heter_feature_2d, record_len, affine_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor  fused_feature
            input data, (B, C, H, W)
        
        heter_feature_2d : torch.Tensor
        input data, (sum(n_cav), C, 2*H, 2*W)
            
        record_len : list
            shape: (B)
            
        affine_matrix : torch.Tensor
            normalized affine matrix from 'normalize_pairwise_tfm'
            shape: (B, L, L, 2, 3) 
            
        """
        # feature map 压缩
        heter_feature_2d = self.relu(self.conv_layer_downsample(heter_feature_2d))

        # 1. Regroup heter_feature_2d
        _, C, H, W = heter_feature_2d.shape
        B, L = affine_matrix.shape[:2]       # 这里L是最大数量6，具体的邻居车辆需要根据record_len确定，无需额外考虑

        regroup_feature, mask = Regroup(heter_feature_2d, record_len, L)
        regroup_feature_new = [] 
        for b in range(B):
            ego = 0
            regroup_feature_new.append(warp_affine_simple(regroup_feature[b], affine_matrix[b, ego], (H, W)))
        regroup_feature = torch.stack(regroup_feature_new) # [B, L, C, H, W]

        # regroup_feature 加入x，维度变为 [B, L+1, C, H, W], 再变为[B*(L+1), C, H, W]
        # regroup_feature = torch.cat([x.unsqueeze(1), regroup_feature], dim=1).reshape(-1, C, H, W)

        # regroup_feature 加入x，维度变为 [B, L+1, C, H, W]
        regroup_feature = torch.cat([x.unsqueeze(1), regroup_feature], dim=1)

        
        # 2. Diffusion


        #将H W从 100 252 padding到 128 256  #dair
        regroup_feature = F.pad(regroup_feature, (2, 2, 2, 2), "constant", 0)
        x = F.pad(x, (2, 2, 2, 2), "constant", 0) 

        # opv2v, v2xset
        # regroup_feature = F.pad(regroup_feature, (0, 0, 2, 2), "constant", 0)
        # x = F.pad(x, (0, 0, 2, 2), "constant", 0)  

        loss = self.diffusion(
            img=x,
            condition=regroup_feature
        )

        return loss
 


if __name__ == '__main__':
    # test

    # # 创建模拟的输入数据
    input_tensor = torch.randn(8, 8, 100, 252)  # 形状为[batch size, channels, height, width]
    padded_tensor = F.pad(input_tensor, (2, 3, 2, 2))
    # print(padded_tensor.shape)

