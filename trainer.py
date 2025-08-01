import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch import amp
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from utils import show_mask
import monai
import pandas as pd
from evaluations.metrics import SegmentationMetrics, ClassificationMetrics, bone_name_dict
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import ConfusionMatrixDisplay
import time
from typing import Tuple
from utils import AdaptiveLossBalancer, LossBalancer
from evaluations.metrics import overlap_pairs


@torch.no_grad()
def _prep_batch(batch, device):
    img, gt = batch["img"], batch["gt"]
    gt = (gt > 0.5).float()  # 二值化
    return img.to(device), gt.to(device)


def mix_pred_gt(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mode: str = "ratio",
    p: float = 0.5,
    k: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    随机把 pred 的部分通道换成 gt。

    参数
    ----
    pred: (B, 14, H, W)  生成器输出
    gt:   (B, 14, H, W)  对应 GT
    mode: "ratio" 或 "fixed"
        - ratio: 每个通道独立以概率 p 替换
        - fixed: 每个样本固定替换 k 个通道
    p:    ratio 模式下替换概率
    k:    fixed 模式下每样本替换通道数
    返回
    ----
    mixed:  (B, 14, H, W)   混合后的输入
    labels: (B, 14)         真值标记 (1=GT, 0=pred)
    """
    B, C, *_ = pred.shape
    device = pred.device

    if mode == "ratio":
        keep_pred = (torch.rand(B, C, device=device) > p)  # True 保留 pred
    elif mode == "fixed":
        keep_pred = torch.ones(B, C, dtype=torch.bool, device=device)
        for i in range(B):
            idx = torch.randperm(C, device=device)[:k]
            keep_pred[i, idx] = False
    else:
        raise ValueError("mode must be 'ratio' or 'fixed'")

    keep_pred = keep_pred.unsqueeze(-1).unsqueeze(-1).float()  # (B, C, 1, 1)
    mixed = pred * keep_pred + gt * (1 - keep_pred)
    labels = (1 - keep_pred.squeeze(-1).squeeze(-1)).float()   # 1=GT, 0=pred
    return mixed, labels


class GANSegTrainer:
    def __init__(self, args, G, D, train_loader, val_loader, criterion_G, criterion_D, optimizer_G, optimizer_D, device="cuda:0"):
        self.G = G
        self.D = D
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion_G = criterion_G
        self.criterion_D = criterion_D
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.amp = args.amp
        self.grad_clip = args.grad_clip
        self.device = device
        self.max_epoch = args.max_epoch
        self.dice_metric = monai.metrics.DiceMetric(reduction="none", ignore_empty=False)
        self.scaler = GradScaler() if self.amp else None
        self.earlystop = EarlyStopping(patience=30)
        self.balancer = AdaptiveLossBalancer(beta=0.99, min_w=0.01, max_w=30.0)

        if args.scheduler == "CosineAnnealing":
            self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, self.max_epoch, eta_min=self.optimizer_G.param_groups[0]['lr'] * 0.01)
            self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, self.max_epoch, eta_min=self.optimizer_D.param_groups[0]['lr'] * 0.01)

        elif args.scheduler == "Plateau":
            self.scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G, mode='min', factor=0.8, patience=5, cooldown=2)
            self.scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_D, mode='min', factor=0.8, patience=5, cooldown=2)

        else:
            self.scheduler_G = None
            self.scheduler_D = None

    def fit(self, args):
        G_train_loss = []
        ADV_train_loss = []
        D_train_loss = []
        val_loss = []
        best_val_loss = np.Inf
        for epoch in range(self.max_epoch):
            if next(self.G.parameters()).device != self.device:
                self.G = self.G.to(self.device)
            if next(self.D.parameters()).device != self.device:
                self.D = self.D.to(self.device)
            if self.amp:
                epoch_G_train_loss_reduced, epoch_ADV_train_loss_reduced, epoch_D_train_loss_reduced = self.train_one_epoch_amp(epoch)
            else:
                epoch_G_train_loss_reduced, epoch_ADV_train_loss_reduced, epoch_D_train_loss_reduced = self.train_one_epoch(epoch)
            G_train_loss.append(epoch_G_train_loss_reduced)
            ADV_train_loss.append(epoch_ADV_train_loss_reduced)
            D_train_loss.append(epoch_D_train_loss_reduced)
            epoch_val_loss_reduced = self.validate(epoch)
            if self.earlystop(epoch_val_loss_reduced):
                break
            val_loss.append(epoch_val_loss_reduced)
            self.plot(args, G_train_loss, ADV_train_loss, D_train_loss, val_loss)
            if args.scheduler == "CosineAnnealing":
                self.scheduler_G.step()
                self.scheduler_D.step()
            elif args.scheduler == "Plateau":
                self.scheduler_G.step(epoch_val_loss_reduced)
                self.scheduler_D.step(epoch_val_loss_reduced)
            ckpt = {
                "model": self.G.state_dict(),
                "dis": self.D.state_dict(),
                "epoch": epoch,
                "optimizer": self.optimizer_G.state_dict(),
                "optimizer_D": self.optimizer_D.state_dict(),
                "train_loss": epoch_G_train_loss_reduced,
                "train_loss_D": epoch_D_train_loss_reduced,
                "val_loss": epoch_val_loss_reduced,
            }
            if epoch_val_loss_reduced < best_val_loss:
                torch.save(ckpt, (Path(args.model_save_path) / "model_best.pth"))
                print(f"New best val loss: {best_val_loss:.4f} -> {epoch_val_loss_reduced:.4f}")
                best_val_loss = epoch_val_loss_reduced
            else:
                torch.save(ckpt, (Path(args.model_save_path) / "model_latest.pth"))
                print(f"Best val_loss didn't decrease, current val_loss: {epoch_val_loss_reduced:.4f}, best val_loss: {best_val_loss:.4f}")

    def train_one_epoch_amp(self, epoch: int):
        """One epoch GAN training with AMP."""
        self.G.train()
        self.D.train()

        pbar = tqdm(self.train_loader, ncols=0)
        avg_g, avg_adv, avg_d = 0.0, 0.0, 0.0

        for step, batch in enumerate(pbar):
            img, gt = _prep_batch(batch, self.device)

            # ==========================================================
            # 1) Generator forward (retain graph) ----------------------
            # ==========================================================
            with autocast():
                # pred, final_pred = self.G(img)  # (B,14,H,W)*2
                final_pred = self.G(img)

            # ==========================================================
            # 2) === 更新 Discriminator ===============================
            #    先用 detach(pred) 混合真 / 假通道，训练 D
            # ==========================================================
            # if epoch < 30:
            #     mixed, label_mask = mix_pred_gt(pred[:, :14, ...].detach(), gt[:, :14, ...],
            #                                     mode="ratio", p=0.5)
            # else:
            # overlap_mul_gt = pred[:, 14:, ...].detach()*gt[:, 14:, ...]
            # final_pred_mul_gt = final_pred.detach()*gt[:, :14, ...]
            # mixed, label_mask = mix_pred_gt(torch.cat([final_pred_mul_gt, overlap_mul_gt], dim=1), gt, mode="ratio", p=0.5)
            mixed, label_mask = mix_pred_gt(final_pred.detach()*gt[:, :14, ...], gt[:, :14, ...], mode="ratio", p=0.5)

            with autocast():
                d_logits = self.D(mixed)  # (B,14)
                d_loss = self.criterion_D(d_logits, label_mask)

            self.scaler.scale(d_loss).backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.D.parameters(),
                                               self.grad_clip)
            self.scaler.step(self.optimizer_D)
            self.scaler.update()
            self.optimizer_D.zero_grad(set_to_none=True)

            # ==========================================================
            # 3) === 更新 Generator ====================================
            #    分割主损失 + 对抗损失，训练 G
            # ==========================================================
            # seg_w = 0.1 if epoch < 30 else 1.0  # 你的阶段权重
            # d_w = 5.0 if epoch < 30 else 20.0
            final_w = 1.0
            with autocast():
                # seg_loss = self.criterion_G(pred, final_pred, gt, final_w)
                seg_loss = self.criterion_G(None, final_pred, gt, final_w)
                # if epoch < 30:
                #     adv_logits = self.D(pred[:, :14, ...])  # 不 detach
                # else:
                # d_in = torch.cat([final_pred, pred[:, 14:, ...]], dim=1)
                # adv_logits = self.D(d_in)
                adv_logits = self.D(final_pred)  # 不 detach
                adv_loss = self.criterion_D(adv_logits, torch.ones_like(adv_logits))
                seg_w, adv_w, d_w = self.balancer.update(
                    seg_loss, adv_loss, d_loss, anchor=final_pred
                )
                # seg_w, adv_w = self.balancer.update(
                #     seg_loss, adv_loss
                # )
                g_loss = seg_w * seg_loss + adv_w * adv_loss

            self.scaler.scale(g_loss).backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.G.parameters(),
                                               self.grad_clip)
            self.scaler.step(self.optimizer_G)
            self.scaler.update()
            self.optimizer_G.zero_grad(set_to_none=True)

            # ==========================================================
            # 4) 统计与日志 --------------------------------------------
            # ==========================================================
            avg_g += g_loss.item()
            avg_adv += adv_loss.item()
            avg_d += d_loss.item()
            pbar.set_description(
                f"Epoch {epoch}  "
                f"G:{g_loss.item():.4f}  ADV: {adv_loss.item():.4f}  D:{d_loss.item():.4f}  "
                f"lrG:{self.optimizer_G.param_groups[0]['lr']:.2e}  "
                f"seg_w: {seg_w} adv_w: {adv_w}"
            )

        avg_g /= len(self.train_loader)
        avg_adv /= len(self.train_loader)
        avg_d /= len(self.train_loader)
        return avg_g, avg_adv, avg_d

    def train_one_epoch(self, epoch: int):
        """One-epoch GAN training."""
        self.G.train()
        self.D.train()

        pbar = tqdm(self.train_loader, ncols=0)
        avg_g, avg_adv, avg_d = 0.0, 0.0, 0.0

        for step, batch in enumerate(pbar):
            # ─────────────────────────────────────────────────────────────
            # 1) 取数据
            # ─────────────────────────────────────────────────────────────
            img, gt = _prep_batch(batch, self.device)

            # ─────────────────────────────────────────────────────────────
            # 2) Generator 前向
            # ─────────────────────────────────────────────────────────────
            final_pred = self.G(img)  # (B, 14, H, W)

            # ─────────────────────────────────────────────────────────────
            # 3) 更新 Discriminator
            #    用 detach(final_pred) 与真值混合，计算 D 损失
            # ─────────────────────────────────────────────────────────────
            mixed, label_mask = mix_pred_gt(
                final_pred.detach() * gt[:, :14, ...],  # 预测 × GT，只取骨类通道
                gt[:, :14, ...],
                mode="ratio",
                p=0.5
            )

            d_logits = self.D(mixed)  # (B, 14)
            d_loss = self.criterion_D(d_logits, label_mask)

            d_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.D.parameters(),
                                               self.grad_clip)
            self.optimizer_D.step()
            self.optimizer_D.zero_grad(set_to_none=True)

            # ─────────────────────────────────────────────────────────────
            # 4) 更新 Generator
            #    = 分割损失 + 对抗损失
            # ─────────────────────────────────────────────────────────────
            seg_loss = self.criterion_G(None, final_pred, gt, final_w=1.0)

            adv_logits = self.D(final_pred)  # 不 detach
            adv_loss = self.criterion_D(
                adv_logits, torch.ones_like(adv_logits)
            )

            seg_w, adv_w, d_w = self.balancer.update(
                seg_loss, adv_loss, d_loss, anchor=final_pred
            )
            g_loss = seg_w * seg_loss + adv_w * adv_loss

            g_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.G.parameters(),
                                               self.grad_clip)
            self.optimizer_G.step()
            self.optimizer_G.zero_grad(set_to_none=True)

            # ─────────────────────────────────────────────────────────────
            # 5) 统计与日志
            # ─────────────────────────────────────────────────────────────
            avg_g += g_loss.item()
            avg_adv += adv_loss.item()
            avg_d += d_loss.item()
            pbar.set_description(
                f"Epoch {epoch}  "
                f"G:{g_loss.item():.4f}  ADV:{adv_loss.item():.4f}  D:{d_loss.item():.4f}  "
                f"lrG:{self.optimizer_G.param_groups[0]['lr']:.2e}  "
                f"seg_w:{seg_w:.3f}  adv_w:{adv_w:.3f}"
            )

        avg_g /= len(self.train_loader)
        avg_adv /= len(self.train_loader)
        avg_d /= len(self.train_loader)
        return avg_g, avg_adv, avg_d

    def validate(self, epoch):
        self.G.eval()
        dice_scores = []
        pbar = tqdm(self.val_loader)
        avg_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(pbar):
                img = batch["img"]
                gt = batch["gt"]
                # Avoid non-binary value caused by resize
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                if img.device != self.device:
                    img = img.to(self.device)
                if gt.device != self.device:
                    gt = gt.to(self.device)
                pred, final_pred = self.G(img)
                loss = self.criterion_G(pred, final_pred, gt, 1)
                pred_bin = final_pred
                pred_bin[pred_bin > 0.5] = 1
                pred_bin[pred_bin <= 0.5] = 0
                dice_score_single = self.dice_metric(pred_bin, gt).squeeze().cpu().numpy().mean()
                dice_scores.append(dice_score_single)
                avg_loss += loss.item()
                pbar.set_description(f"Epoch {epoch} Validating at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                                     f"loss: {loss.item():.4f}, lr:{self.optimizer_G.param_groups[0]['lr']}")
        avg_loss /= len(self.val_loader)
        dice_score_reduced = np.array(dice_scores).mean()
        print("Dice: ", dice_score_reduced)
        return avg_loss

    def plot(self, args, G_train_loss, ADV_train_loss, D_train_loss, val_loss):
        plt.plot(G_train_loss, label='G Train Loss')
        plt.plot(ADV_train_loss, label='ADV Train Loss')
        plt.plot(D_train_loss, label='D Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(Path(args.model_save_path) / "loss_curve.png")
        plt.close()


class DualSegTrainer:
    def __init__(self, args, net, train_loader, val_loader, criterion, optimizer, device="cuda:0"):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.overlap_consistency = monai.losses.FocalLoss(include_background=True)
        self.optimizer = optimizer
        self.amp = args.amp
        self.grad_clip = args.grad_clip
        self.device = device
        self.max_epoch = args.max_epoch
        self.dice_metric = monai.metrics.DiceMetric(reduction="none", ignore_empty=False)
        self.scaler = GradScaler() if self.amp else None
        self.earlystop = EarlyStopping(patience=10)
        if args.scheduler == "CosineAnnealing":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch, eta_min=self.optimizer.param_groups[0]['lr'] * 0.01)
        elif args.scheduler == "Plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=5, cooldown=2)
        else:
            self.scheduler = None

    def fit(self, args):
        train_loss = []
        val_loss = []
        best_val_loss = np.Inf
        for epoch in range(self.max_epoch):
            if next(self.net.parameters()).device != self.device:
                self.net = self.net.to(self.device)
            if self.amp:
                epoch_train_loss_reduced = self.train_one_epoch_amp(epoch)
            else:
                epoch_train_loss_reduced = self.train_one_epoch(epoch)
            train_loss.append(epoch_train_loss_reduced)
            epoch_val_loss_reduced = self.validate(epoch)
            if self.earlystop(epoch_val_loss_reduced):
                break
            val_loss.append(epoch_val_loss_reduced)
            self.plot(args, train_loss, val_loss)
            if args.scheduler == "CosineAnnealing":
                self.scheduler.step()
            elif args.scheduler == "Plateau":
                self.scheduler.step(epoch_val_loss_reduced)
            ckpt = {
                "model": self.net.state_dict(),
                "epoch": epoch,
                "optimizer": self.optimizer.state_dict(),
                "train_loss": epoch_train_loss_reduced,
                "val_loss": epoch_val_loss_reduced,
            }
            if epoch_val_loss_reduced < best_val_loss:
                torch.save(ckpt, (Path(args.model_save_path) / "model_best.pth"))
                print(f"New best val loss: {best_val_loss:.4f} -> {epoch_val_loss_reduced:.4f}")
                best_val_loss = epoch_val_loss_reduced
            else:
                torch.save(ckpt, (Path(args.model_save_path) / "model_latest.pth"))
                print(f"Best val_loss didn't decrease, current val_loss: {epoch_val_loss_reduced:.4f}, best val_loss: {best_val_loss:.4f}")

    def train_one_epoch_amp(self, epoch):
        self.net.train()
        pbar = tqdm(self.train_loader)
        avg_loss = 0
        for step, batch in enumerate(pbar):
            img = batch["img"]
            gt = batch["gt"]
            # Avoid non-binary value caused by resize
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if img.device != self.device:
                img = img.to(self.device)
            if gt.device != self.device:
                gt = gt.to(self.device)
            with autocast():
                pred, final_pred = self.net(img)
                if epoch < 30:
                    loss = self.criterion(pred, final_pred, gt, 0.1)
                else:
                    loss = self.criterion(pred, final_pred, gt, 1)
            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            avg_loss += loss.item()
            pbar.set_description(f"Epoch {epoch} training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                                 f"loss: {loss.item():.4f}, lr:{self.optimizer.param_groups[0]['lr']}")
        avg_loss /= len(self.train_loader)
        return avg_loss

    def train_one_epoch(self, epoch):
        """One-epoch training without mixed precision."""
        self.net.train()
        pbar = tqdm(self.train_loader)
        avg_loss = 0.0

        for step, batch in enumerate(pbar):
            img, gt = batch["img"], batch["gt"]

            # 二值化标签，避免插值带来的非零非一值
            gt = (gt > 0.5).float()

            img = img.to(self.device, non_blocking=True)
            gt = gt.to(self.device, non_blocking=True)

            # 前向传播
            pred, final_pred = self.net(img)
            weight = 0.1 if epoch < 30 else 1.0
            loss = self.criterion(pred, final_pred, gt, weight)

            # 反向传播
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # 日志
            avg_loss += loss.item()
            pbar.set_description(
                f"Epoch {epoch} training at {datetime.now():%Y-%m-%d %H:%M:%S}, "
                f"loss: {loss.item():.4f}, lr:{self.optimizer.param_groups[0]['lr']}"
            )

        avg_loss /= len(self.train_loader)
        return avg_loss

    def validate(self, epoch):
        self.net.eval()
        dice_scores = []
        pbar = tqdm(self.val_loader)
        avg_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(pbar):
                img = batch["img"]
                gt = batch["gt"]
                # Avoid non-binary value caused by resize
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                if img.device != self.device:
                    img = img.to(self.device)
                if gt.device != self.device:
                    gt = gt.to(self.device)
                pred, final_pred = self.net(img)
                loss = self.criterion(pred, final_pred, gt, 1)
                pred_bin = final_pred
                pred_bin[pred_bin > 0.5] = 1
                pred_bin[pred_bin <= 0.5] = 0

                dice_score_single = self.dice_metric(pred_bin, gt).squeeze().cpu().numpy().mean()
                dice_scores.append(dice_score_single)
                avg_loss += loss.item()
                pbar.set_description(f"Epoch {epoch} Validating at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                                     f"loss: {loss.item():.4f}, lr:{self.optimizer.param_groups[0]['lr']}")
        avg_loss /= len(self.val_loader)
        dice_score_reduced = np.array(dice_scores).mean()
        print("Dice: ", dice_score_reduced)
        return avg_loss

    def plot(self, args, train_loss, val_loss):
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(Path(args.model_save_path) / "loss_curve.png")
        plt.close()


class DualSegTester:
    def __init__(self, args, net, test_loader, device="cuda:0"):
        self.args = args
        self.net = net
        self.net.load_state_dict(torch.load((Path(args.checkpoint) / "model_best.pth"))["model"])
        self.test_loader = test_loader
        self.device = device
        self.save_overlay = args.save_overlay
        self.save_csv = args.save_csv
        self.save_pred = args.save_pred
        self.colors = [
            [0.1522, 0.4717, 0.9685],
            [0.3178, 0.0520, 0.8333],
            [0.3834, 0.3823, 0.6784],
            [0.8525, 0.1303, 0.4139],
            [0.9948, 0.8252, 0.3384],
            [0.8476, 0.7147, 0.2453],
            [0.2865, 0.8411, 0.0877],
            [0.1558, 0.4940, 0.4668],
            [0.9199, 0.5882, 0.5113],
            [0.1335, 0.5433, 0.6149],
            [0.0629, 0.7343, 0.0943],
            [0.8183, 0.2786, 0.3053],
            [0.1789, 0.5083, 0.6787],
            [0.9746, 0.1909, 0.4295],
            [0.1586, 0.8670, 0.6994],
            [0.9156, 0.1241, 0.3829],
            [0.2998, 0.3054, 0.4242],
            [0.7719, 0.7786, 0.1164],
            [0.8033, 0.9278, 0.7621],
            [0.1085, 0.5155, 0.4145]
        ]
        self.metrics = SegmentationMetrics(num_classes=14)
        # self.Precision = monai.metrics.
        if next(self.net.parameters()).device != self.device:
            self.net = self.net.to(self.device)

    def test(self):
        self.net.eval()
        pbar = tqdm(self.test_loader)
        total_infer_time = 0
        total_items = 0
        with torch.no_grad():
            for step, batch in enumerate(pbar):
                img = batch["img"]
                gt = batch["gt"]
                # Avoid non-binary value caused by resize
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                if img.device != self.device:
                    img = img.to(self.device)
                if gt.device != self.device:
                    gt = gt.to(self.device)
                start_time = time.time()  # ⏱️ Start timing
                _, pred = self.net(img)
                end_time = time.time()  # ⏱️ End timing
                infer_time = end_time - start_time
                total_infer_time += infer_time
                total_items += img.shape[0]  # 批量大小
                pred_bin = pred
                pred_bin[pred_bin > 0.5] = 1
                pred_bin[pred_bin <= 0.5] = 0

                self.metrics.update_metrics(pred_bin, gt[:, :14, ...], batch["fname"][0])
                pbar.set_description(f"Testing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                if self.save_overlay:
                    self.create_overlay(self.args, image=img, pred=pred[:, :14, ...], mask=gt[:, :14, ...], fname=batch["fname"])
                if self.save_pred:
                    self.create_pred(self.args, image=img, pred=pred[:, :14, ...], mask=gt[:, :14, ...], fname=batch["fname"])
                    self.create_overlay_single(self.args, image=img, pred=pred[:, :14, ...], mask=gt[:, :14, ...], fname=batch["fname"])
        if self.save_csv:
            self.create_csv(self.args)
        metrics_dict = self.metrics.get_metrics()
        dsc_reduced = metrics_dict["dsc"].mean()
        print("Mean DSC: ", dsc_reduced)
        nsd_reduced = metrics_dict["nsd"].mean()
        print("Mean NSD: ", nsd_reduced)

        avg_infer_time = total_infer_time / total_items
        print(f"Average inference time per item: {avg_infer_time * 1000:.2f} ms")

    def create_overlay(self, args, image, pred, mask, fname):
        save_path = Path(args.checkpoint) / "overlay"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        pred_mask_bin = pred.detach()
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        ax[0].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[1].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[2].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[0].set_title("Image")
        ax[1].set_title("Segmentation")
        ax[2].set_title("GT")
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')

        for i in range(pred_mask_bin.shape[1]):
            seg = pred_mask_bin[0][i].cpu().numpy()
            show_mask((seg == 1).astype(np.uint8), ax[1], mask_color=np.array(self.colors[i]))
            show_mask((mask[0][i].cpu().numpy() == 1).astype(np.uint8), ax[2], mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '.pdf'), dpi=600)
        plt.close()

    def create_pred(self, args, image, pred, mask, fname):
        save_path = Path(args.checkpoint) / "pred"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        pred_mask_bin = pred.detach()
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0

        fig_pred, ax_pred = plt.subplots(figsize=(5, 5))
        ax_pred.axis('off')  # 不显示坐标轴
        for i in range(pred_mask_bin.shape[1]):
            seg = pred_mask_bin[0][i].cpu().numpy()
            show_mask((seg == 1).astype(np.uint8), ax_pred, mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '_pred.pdf'), dpi=600)
        plt.close()

        fig_gt, ax_gt = plt.subplots(figsize=(5, 5))
        ax_gt.axis('off')  # 不显示坐标轴
        for i in range(pred_mask_bin.shape[1]):
            show_mask((mask[0][i].cpu().numpy() == 1).astype(np.uint8), ax_gt, mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '_gt.pdf'), dpi=600)
        plt.close()

    def create_overlay_single(self, args, image, pred, mask, fname):
        save_path = Path(args.checkpoint) / "overlay_single"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        pred_mask_bin = pred.detach()
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0

        fig_pred, ax_pred = plt.subplots(figsize=(5, 5))
        ax_pred.imshow(image[0][0].cpu().numpy(), 'gray')
        ax_pred.axis('off')  # 不显示坐标轴
        for i in range(pred_mask_bin.shape[1]):
            seg = pred_mask_bin[0][i].cpu().numpy()
            show_mask((seg == 1).astype(np.uint8), ax_pred, mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '_pred.pdf'), dpi=600)
        plt.close()

        fig_gt, ax_gt = plt.subplots(figsize=(5, 5))
        ax_gt.imshow(image[0][0].cpu().numpy(), 'gray')
        ax_gt.axis('off')  # 不显示坐标轴
        for i in range(pred_mask_bin.shape[1]):
            show_mask((mask[0][i].cpu().numpy() == 1).astype(np.uint8), ax_gt, mask_color=np.array(self.colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '_gt.pdf'), dpi=600)
        plt.close()

    def create_csv(self, args):
        save_path = Path(args.checkpoint)
        metrics_dict = self.metrics.get_metrics()
        num_classes = self.metrics.num_labels
        overlap_dsc_mean_df = pd.DataFrame(metrics_dict["overlap_dsc"], columns=["Mean Overlap DSC"])
        overlap_dsc_df = pd.DataFrame(metrics_dict["overlap_dsc_per_pair"], columns=[f"Overlap DSC {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for pair in metrics_dict["overlap_pairs"]])
        overlap_nsd_mean_df = pd.DataFrame(metrics_dict["overlap_nsd"], columns=["Mean Overlap NSD"])
        overlap_nsd_df = pd.DataFrame(metrics_dict["overlap_nsd_per_pair"], columns=[f"Overlap NSD {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for pair in metrics_dict["overlap_pairs"]])
        overlap_voe_mean_df = pd.DataFrame(metrics_dict["overlap_voe"], columns=["Mean Overlap VOE"])
        overlap_voe_df = pd.DataFrame(metrics_dict["overlap_voe_per_pair"], columns=[f"Overlap VOE {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for pair in metrics_dict["overlap_pairs"]])
        overlap_msd_mean_df = pd.DataFrame(metrics_dict["overlap_msd"], columns=["Mean Overlap MSD"])
        overlap_msd_df = pd.DataFrame(metrics_dict["overlap_msd_per_pair"], columns=[f"Overlap MSD {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for pair in metrics_dict["overlap_pairs"]])
        overlap_ravd_mean_df = pd.DataFrame(metrics_dict["overlap_ravd"], columns=["Mean Overlap RAVD"])
        overlap_ravd_df = pd.DataFrame(metrics_dict["overlap_ravd_per_pair"], columns=[f"Overlap RAVD {bone_name_dict[pair[0]]}-{bone_name_dict[pair[1]]}" for pair in metrics_dict["overlap_pairs"]])

        dsc_df = pd.DataFrame(metrics_dict["dsc_pc"], columns=[f"DSC {bone_name_dict[i]}" for i in range(num_classes)])
        dsc_mean_df = pd.DataFrame(metrics_dict["dsc"], columns=["Mean DSC"])
        nsd_df = pd.DataFrame(metrics_dict["nsd_pc"], columns=[f"NSD {bone_name_dict[i]}" for i in range(num_classes)])
        nsd_mean_df = pd.DataFrame(metrics_dict["nsd"], columns=["Mean NSD"])
        voe_df = pd.DataFrame(metrics_dict["voe_pc"], columns=[f"VOE {bone_name_dict[i]}" for i in range(num_classes)])
        voe_mean_df = pd.DataFrame(metrics_dict["voe"], columns=["Mean VOE"])
        msd_df = pd.DataFrame(metrics_dict["msd_pc"], columns=[f"MSD {bone_name_dict[i]}" for i in range(num_classes)])
        msd_mean_df = pd.DataFrame(metrics_dict["msd"], columns=["Mean MSD"])
        ravd_df = pd.DataFrame(metrics_dict["ravd_pc"], columns=[f"RAVD {bone_name_dict[i]}" for i in range(num_classes)])
        ravd_mean_df = pd.DataFrame(metrics_dict["ravd"], columns=["Mean RAVD"])

        fname_df = pd.DataFrame(metrics_dict["fname"], columns=['Case'])
        metric_df = pd.concat(
         [fname_df, overlap_dsc_df, overlap_dsc_mean_df, overlap_nsd_df, overlap_nsd_mean_df, overlap_voe_df,
               overlap_voe_mean_df, overlap_msd_df, overlap_msd_mean_df, overlap_ravd_df, overlap_ravd_mean_df, dsc_df,
               dsc_mean_df, nsd_df, nsd_mean_df, voe_df, voe_mean_df, msd_df, msd_mean_df, ravd_df, ravd_mean_df], axis=1)
        column_means = metric_df.iloc[:, 1:].mean()
        average_row = pd.DataFrame([['Average'] + column_means.tolist()], columns=metric_df.columns)
        final_df = pd.concat([metric_df, average_row], ignore_index=True)
        final_df.to_csv((save_path / 'test_metrics.csv'), index=False)


class SegInferer:
    def __init__(self, args, net, test_loader, device="cuda:0"):
        self.args = args
        self.net = net
        self.net.load_state_dict(torch.load((Path(args.checkpoint) / "model_best.pth"))["model"])
        self.test_loader = test_loader
        self.device = device
        self.save_overlay = args.save_overlay
        self.save_csv = args.save_csv
        self.metrics = SegmentationMetrics(num_classes=14)
        if next(self.net.parameters()).device != self.device:
            self.net = self.net.to(self.device)

    def test(self):
        self.net.eval()
        pbar = tqdm(self.test_loader)
        with torch.no_grad():
            for step, batch in enumerate(pbar):
                img = batch["img"]
                gt = batch["gt"]
                # Avoid non-binary value caused by resize
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                if img.device != self.device:
                    img = img.to(self.device)
                if gt.device != self.device:
                    gt = gt.to(self.device)
                pred = self.net(img)
                pred_bin = pred
                pred_bin[pred_bin > 0.5] = 1
                pred_bin[pred_bin <= 0.5] = 0
                self.metrics.update_metrics(pred_bin, gt, batch["fname"][0])
                pbar.set_description(f"Testing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                if self.save_overlay:
                    self.create_overlay(self.args, image=img, pred=pred, mask=gt, fname=batch["fname"])
        if self.save_csv:
            self.create_csv(self.args)
        metrics_dict = self.metrics.get_metrics()
        dsc_reduced = metrics_dict["dsc"].mean()
        print("Mean DSC: ", dsc_reduced)
        nsd_reduced = metrics_dict["nsd"].mean()
        print("Mean NSD: ", nsd_reduced)

    def create_overlay(self, args, image, pred, mask, fname):
        save_path = Path(args.checkpoint) / "overlay"
        if not save_path.exists():
            save_path.mkdir(parents=True)
        colors = [
            [0.1522, 0.4717, 0.9685],
            [0.3178, 0.0520, 0.8333],
            [0.3834, 0.3823, 0.6784],
            [0.8525, 0.1303, 0.4139],
            [0.9948, 0.8252, 0.3384],
            [0.8476, 0.7147, 0.2453],
            [0.2865, 0.8411, 0.0877],
            [0.1558, 0.4940, 0.4668],
            [0.9199, 0.5882, 0.5113],
            [0.1335, 0.5433, 0.6149],
            [0.0629, 0.7343, 0.0943],
            [0.8183, 0.2786, 0.3053],
            [0.1789, 0.5083, 0.6787],
            [0.9746, 0.1909, 0.4295],
            [0.1586, 0.8670, 0.6994],
            [0.9156, 0.1241, 0.3829],
            [0.2998, 0.3054, 0.4242],
            [0.7719, 0.7786, 0.1164],
            [0.8033, 0.9278, 0.7621],
            [0.1085, 0.5155, 0.4145]
        ]
        pred_mask_bin = pred.detach()
        pred_mask_bin[pred_mask_bin > 0.5] = 1
        pred_mask_bin[pred_mask_bin <= 0.5] = 0
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        ax[0].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[1].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[2].imshow(image[0][0].cpu().numpy(), 'gray')
        ax[0].set_title("Image")
        ax[1].set_title("Segmentation")
        ax[2].set_title("GT")
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')

        for i in range(pred_mask_bin.shape[1]):
            seg = pred_mask_bin[0][i].cpu().numpy()
            show_mask((seg == 1).astype(np.uint8), ax[1], mask_color=np.array(colors[i]))
            show_mask((mask[0][i].cpu().numpy() == 1).astype(np.uint8), ax[2], mask_color=np.array(colors[i]))
        plt.tight_layout()
        plt.savefig(save_path / (fname[0] + '.pdf'), dpi=600)
        plt.close()

    def create_csv(self, args):
        save_path = Path(args.checkpoint)
        metrics_dict = self.metrics.get_metrics()
        num_classes = self.metrics.num_labels
        dsc_df = pd.DataFrame(metrics_dict["dsc_pc"], columns=[f"DSC {bone_name_dict[i]}" for i in range(num_classes)])
        dsc_mean_df = pd.DataFrame(metrics_dict["dsc"], columns=["Mean DSC"])
        nsd_df = pd.DataFrame(metrics_dict["nsd_pc"], columns=[f"NSD {bone_name_dict[i]}" for i in range(num_classes)])
        nsd_mean_df = pd.DataFrame(metrics_dict["nsd"], columns=["Mean NSD"])
        voe_df = pd.DataFrame(metrics_dict["voe_pc"], columns=[f"VOE {bone_name_dict[i]}" for i in range(num_classes)])
        voe_mean_df = pd.DataFrame(metrics_dict["voe"], columns=["Mean VOE"])
        msd_df = pd.DataFrame(metrics_dict["msd_pc"], columns=[f"MSD {bone_name_dict[i]}" for i in range(num_classes)])
        msd_mean_df = pd.DataFrame(metrics_dict["msd"], columns=["Mean MSD"])
        ravd_df = pd.DataFrame(metrics_dict["ravd_pc"], columns=[f"RAVD {bone_name_dict[i]}" for i in range(num_classes)])
        ravd_mean_df = pd.DataFrame(metrics_dict["ravd"], columns=["Mean RAVD"])


        fname_df = pd.DataFrame(metrics_dict["fname"], columns=['Case'])
        metric_df = pd.concat(
         [fname_df, dsc_df, dsc_mean_df, nsd_df, nsd_mean_df, voe_df, voe_mean_df, msd_df, msd_mean_df,
               ravd_df, ravd_mean_df], axis=1)

        column_means = metric_df.iloc[:, 1:].mean()
        average_row = pd.DataFrame([['Average'] + column_means.tolist()], columns=metric_df.columns)
        final_df = pd.concat([metric_df, average_row], ignore_index=True)
        final_df.to_csv((save_path / 'test_metrics.csv'), index=False)


class EarlyStopping:
    def __init__(self, patience, delta=0.0, mode="min"):
        self.patience = patience
        self.mode = mode
        self.delta = 0.0
        self.best_score = None
        self.counter = 0

    def __call__(self, val_metric):
        score = -val_metric if self.mode == 'min' else val_metric
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False
