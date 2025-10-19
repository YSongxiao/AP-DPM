import argparse
import torch
import torch.nn as nn
from utils import *
from trainer import GANSegTrainer, DualSegTester, SegInferer
import monai
import timm
from pathlib import Path
from datasets.carpal import CarpalNpyDatasetWithOverlapInfer, CarpalNpyDatasetWithOverlap, get_dataloader
import torch.optim as optim
from datetime import datetime
from typing import Union
from models.UnetPlusPlus import UnetPlusPlus
from models.SwinUMamba import get_SwinUMamba, get_DPMSwinUMamba


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--seed',
        type=int,
        default=3407,
        help='Seed',
    )

    parser.add_argument(
        '--mode',
        type=str,
        default="train",
        choices=["train", "test", "infer"],
        help='Mode',
    )

    parser.add_argument(
        '--image_size',
        type=int,
        default=512,
        help='The size of the images.',
    )

    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=2,
        help='Batch size for training.',
    )

    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=1,
        help='Batch size for validating.',
    )

    parser.add_argument(
        '--model',
        type=str,
        default="SwinUMamba",
        choices=["DPMSwinUMamba"],
        help='The name of the model.',
    )

    parser.add_argument(
        '--scheduler',
        type=str,
        default="CosineAnnealing",
        choices=["CosineAnnealing", "Plateau"],
        help='The name of the model.',
    )

    parser.add_argument(
        '--amp',
        type=bool,
        default=True,
        help='Whether use amp.',
    )

    parser.add_argument(
        '--grad_clip',
        type=Union[None, float],
        default=None,
        help='Whether use grad_clip.',
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default="",
        help='The path to data.',
    )

    parser.add_argument(
        '--stage1_checkpoint',
        type=str,
        default="/path/to/the/ckpt",
        help='The pretrained weight of the model from stage 1.',
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default="/path/to/the/ckpt",
        help='The pretrained weight of the model.',
    )

    parser.add_argument(
        '--trial_name',
        type=str,
        default="apdpm",
        help='The name of the trial.',
    )

    parser.add_argument(
        '--max_epoch',
        type=int,
        default=100,
        help='Number of epochs.',
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Initial lr',
    )

    parser.add_argument(
        '--save_overlay',
        action='store_true',
        default=False,
        help='Whether to save the overlay (Test mode only).'
    )

    parser.add_argument(
        '--save_csv',
        action='store_true',
        default=False,
        help='Whether save csv (Test mode only).',
    )

    parser.add_argument(
        '--save_pred',
        action='store_true',
        default=True,
        help='Whether save pred (Test mode only).',
    )

    parser.add_argument(
        '--save_npy',
        action='store_true',
        default=False,
        help='Whether to save the npy (Inference mode only).'
    )
    args = parser.parse_args()
    return args


args = get_args()
# Seed everything
seed_everything(args.seed)


# initial network
if args.model == "DPMSwinUMamba":
    net = get_DPMSwinUMamba(in_channels=1, num_classes=14, num_overlap_classes=14)
    if args.mode == "train":
        assert Path(args.stage1_checkpoint).exists(), ("Couldn't load ckpt from Stage 1, please confirm whether you have the "
                                                       "correct path to the ckpt from stage 1.")
        net.load_state_dict(torch.load(args.stage1_checkpoint)["model"])

n_params = sum(p.numel() for p in net.parameters())
print(f"Total parameters: {n_params / 1e6:.2f} M ({n_params:,} parameters)")


class GLossFunction:
    def __init__(self):
        self.loss_fn = monai.losses.DiceLoss(sigmoid=False, squared_pred=True, reduction='mean')
        self.feature_loss_fn = nn.MSELoss()

    def __call__(self, pred, final_pred, gt, final_weight):
        final_dice = self.loss_fn(final_pred, gt[:, :14])
        if pred is not None:
            overall_dice = self.loss_fn(pred[:, :14, ...], gt[:, :14, ...])
            overlap_dice = self.loss_fn(pred[:, 14:, ...], gt[:, 14:, ...])
            loss = overall_dice + overlap_dice + final_weight * final_dice
        else:
            loss = final_dice
        return loss


class DLossFunction:
    def __init__(self):
        self.bce_fn = nn.BCEWithLogitsLoss()

    def __call__(self, pred, gt):
        loss = self.bce_fn(pred, gt)
        return loss


if args.mode == "train":
    # 获取当前时间
    now = datetime.now()
    time_str = now.strftime('%Y%m%d%H%M')
    ckpt_path = Path("./ckpts") / (args.trial_name + "_" + args.model.lower() + "_" + time_str)
    if not ckpt_path.exists():
        ckpt_path.mkdir(parents=True)
    args.model_save_path = str(ckpt_path)
    transform_tr = get_transform(split="train", image_size=args.image_size)
    transform_val = get_transform(split="val", image_size=args.image_size)
    train_dataset = CarpalNpyDatasetWithOverlap(data_root=Path(args.data_path) / "images", annotation_path=Path(args.data_path) / "masks" / "train",
                                                transform=transform_tr)
    train_loader = get_dataloader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_dataset = CarpalNpyDatasetWithOverlap(data_root=Path(args.data_path) / "images", annotation_path=Path(args.data_path) / "masks" / "val",
                                              transform=transform_val)
    val_loader = get_dataloader(val_dataset, batch_size=args.val_batch_size, shuffle=False)
    D = timm.create_model(
        'resnet50d.ra2_in1k',
        #'resnet18d.ra2_in1k',
        pretrained=True,  # 想要预训练就设 True
        in_chans=14,  # 把第一层 conv 改成 1 输入
        num_classes=14
    )
    optimizer_G = optim.AdamW(net.parameters(), lr=args.lr)
    optimizer_D = optim.AdamW(D.parameters(), lr=args.lr*0.25)

    criterion_G = GLossFunction()
    criterion_D = DLossFunction()
    trainer = GANSegTrainer(args, net, D, train_loader, val_loader, criterion_G, criterion_D, optimizer_G, optimizer_D, device="cuda:0")
    trainer.fit(args)

elif args.mode == "test":
    transform_test = get_transform(split="test", image_size=args.image_size)
    test_dataset = CarpalNpyDatasetWithOverlap(data_root=Path(args.data_path) / "images", annotation_path=Path(args.data_path) / "masks" / "test",
                                    transform=transform_test)
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)
    if not (Path(args.checkpoint) / "model_best.pth").exists():
        raise KeyError("Test mode is set but checkpoint does not exist.")
    tester = DualSegTester(args, net, test_loader, device="cuda:0")
    tester.test()

elif args.mode == "infer":
    transform_test = get_transform(split="test", image_size=args.image_size)
    test_dataset = CarpalNpyDatasetWithOverlapInfer(data_root=Path(args.data_path), transform=transform_test)
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)
    if not (Path(args.checkpoint) / "model_best.pth").exists():
        raise KeyError("Test mode is set but checkpoint does not exist.")
    tester = SegInferer(args, net, test_loader, device="cuda:0")
    tester.test()
