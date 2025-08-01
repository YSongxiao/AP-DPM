import random, os
import numpy as np
import torch
import torch.nn as nn
from albumentations import Compose, HorizontalFlip, RandomScale, VerticalFlip, Rotate, Resize, ShiftScaleRotate, RandomBrightnessContrast, CenterCrop
from albumentations.pytorch import ToTensorV2
import cv2
import torch.nn.functional as F
import abc

import torch
import torch.nn as nn
import math
from dataclasses import dataclass

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_transform(split, image_size):
    if split == "train":
        transform = Compose([
            Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            RandomScale(scale_limit=(1.0, 1.1), p=0.5),
            CenterCrop(image_size, image_size),
            ShiftScaleRotate(rotate_limit=10),
            RandomBrightnessContrast(p=0.5),
            ToTensorV2()
        ], additional_targets={'mask': 'gt'})
    elif split == "val" or split == "test":
        transform = Compose([
            Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            ToTensorV2()
        ], additional_targets={'mask': 'gt'})
    else:
        raise NotImplementedError(f"{split} is not implemented.")
    return transform


class NoiseInjection(nn.Module):
    def __init__(self, p: float = 0.0, alpha: float = 0.05):
        super(NoiseInjection, self).__init__()
        self.p = p
        self.alpha = alpha

    def get_noise(self, x):
        dims = tuple(i for i in range(len(x.shape)) if i != 1)
        std = torch.std(x, dim=dims, keepdim=True)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * std
        return noise

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.shape, device=x.device, dtype=x.dtype)
            mask = (mask < self.p).float() * 1
            x = x + self.alpha * mask * self.get_noise(x)
            return x
        return x


class NoiseMultiplicativeInjection(nn.Module):
    def __init__(self, p: float = 0.05, alpha: float = 0.05, betta: float = 0.01):
        super(NoiseMultiplicativeInjection, self).__init__()
        self.p = p
        self.alpha = alpha
        self.betta = betta

    def get_noise(self, x):
        dims = tuple(i for i in range(len(x.shape)) if i != 1)
        std = torch.std(x, dim=dims, keepdim=True)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * std
        return noise

    def get_m_noise(self, x):
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * self.betta + 1
        return noise

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.shape, device=x.device, dtype=x.dtype)
            mask = (mask < self.p).float() * 1
            mask_m = torch.rand(x.shape, device=x.device, dtype=x.dtype)
            mask_m = (mask_m < self.p).float() * 1
            x = x + x * mask_m * self.get_m_noise(x) + self.alpha * mask * self.get_noise(x)
            return x
        return x


@dataclass
class AdaptiveLossBalancer:
    """
    Dynamically adjust (seg_w, adv_w, d_w) to stabilise GAN training.

    Attributes
    ----------
    beta : float
        EMA 衰减系数。β 越大，历史记忆越长。
    min_w, max_w : float
        权重裁剪范围，防止过大过小导致梯度爆炸 / 消失。
    target_ratio : float
        希望保持的 `seg_grad_norm / adv_grad_norm` 目标比例。
    """
    beta: float = 0.99
    min_w: float = 0.1
    max_w: float = 30.0
    target_ratio: float = 1.0  # 想要两支梯度幅度大致相等

    # 以下字段运行时自动维护
    ema_seg: float = 0.0      # ⟨‖∇seg_loss‖⟩_EMA
    ema_adv: float = 0.0      # ⟨‖∇adv_loss‖⟩_EMA
    ema_d:   float = 0.0      # ⟨d_loss⟩_EMA，用于判别器调度

    def update(
        self,
        seg_loss: torch.Tensor,
        adv_loss: torch.Tensor,
        d_loss:   torch.Tensor,
        anchor:   torch.Tensor,   # ← 新增：通常传 final_pred
    ) -> tuple[float, float, float]:

        # 1) 梯度 w.r.t. anchor (retain_graph 保留后向图，稍后还要 .backward())
        seg_grad = torch.autograd.grad(
            seg_loss, anchor, retain_graph=True, create_graph=False
        )[0]
        adv_grad = torch.autograd.grad(
            adv_loss, anchor, retain_graph=True, create_graph=False
        )[0]

        # 2) L2-norm（可换成 mean / max，按需）
        seg_gn = seg_grad.float().norm().item()
        adv_gn = adv_grad.float().norm().item()

        # 2) 更新 EMA
        self.ema_seg = self.beta * self.ema_seg + (1 - self.beta) * seg_gn
        self.ema_adv = self.beta * self.ema_adv + (1 - self.beta) * adv_gn
        self.ema_d   = self.beta * self.ema_d   + (1 - self.beta) * d_loss.item()

        # 3) 计算 Generator 内部两项的权重
        #    想让  seg_w * ema_seg ≈ adv_w * ema_adv * target_ratio
        #    固定 seg_w = 1，求 adv_w.
        if self.ema_adv > 0:
            adv_w = (self.ema_seg / self.ema_adv) * self.target_ratio
        else:
            adv_w = 1.0
        adv_w = float(min(max(adv_w, self.min_w), self.max_w))
        seg_w = 1.0

        # 4) 计算 Discriminator 整体权重（可用于 lr 或 loss 放大因子）
        #    判别器过强 → 减小 d_w；过弱 → 增大 d_w
        #    这里用对数函数确保变化缓和
        #        目标:  <d_loss> ≈ 0.5
        strength = math.exp(self.ema_d - 0.5)   # 0.5 左右最平衡
        d_w = float(min(max(strength, self.min_w), self.max_w))
        # print("[DEBUG]", seg_gn, adv_gn, self.ema_seg, self.ema_adv)
        return seg_w, adv_w, d_w


class LossBalancer:
    def __init__(self, beta=0.99, min_w=0.1, max_w=30.0):
        self.beta, self.min_w, self.max_w = beta, min_w, max_w
        self.es, self.ea = 0., 0.

    def update(self, seg_loss, adv_loss):
        self.es = self.beta * self.es + (1 - self.beta) * seg_loss.detach().item()
        self.ea = self.beta * self.ea + (1 - self.beta) * adv_loss.detach().item()
        ratio   = self.es / (self.ea + 1e-8)      # >1 → seg 比 adv 大
        adv_w   = float(np.clip(ratio, self.min_w, self.max_w))
        seg_w   = 1.0
        return seg_w, adv_w
