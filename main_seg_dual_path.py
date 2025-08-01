import argparse
from utils import *
from trainer import DualSegTrainer, DualSegTester
import monai
from pathlib import Path
from datasets.carpal import CarpalNpyDataset, CarpalNpyDatasetWithOverlap, get_dataloader
import torch.optim as optim
from datetime import datetime
from typing import Union
from models.UnetPlusPlus import UnetPlusPlus
from models.SwinUMamba import get_SwinUMamba, get_DPMSwinUMamba
import segmentation_models_pytorch as smp


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
        default=4,
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
        default="DPMSwinUMamba",
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
        '--checkpoint',
        type=str,
        default="",
        help='The pretrained weight of the model.',
    )

    parser.add_argument(
        '--trial_name',
        type=str,
        default="Train",
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
        default=False,
        help='Whether save pred (Test mode only).',
    )
    args = parser.parse_args()
    return args


args = get_args()
# Seed everything
seed_everything(args.seed)


# initial network
if args.model == "SwinUMamba":
    net = get_SwinUMamba(in_channels=1, num_classes=14)

elif args.model == "DPMSwinUMamba":
    # net = get_DPMSwinUMamba(in_channels=1, num_classes=14, num_overlap_classes=14, feat_size=[64, 128, 256, 512, 1024], hidden_size=1024)
    net = get_DPMSwinUMamba(in_channels=1, num_classes=14, num_overlap_classes=14)


n_params = sum(p.numel() for p in net.parameters())
print(f"Total parameters: {n_params / 1e6:.2f} M ({n_params:,} parameters)")


class MyLossFunction:
    def __init__(self):
        self.loss_fn = monai.losses.DiceLoss(sigmoid=False, squared_pred=True, reduction='mean')

    def __call__(self, pred, final_pred, gt, final_weight):
        overall_dice = self.loss_fn(pred[:, :14, ...], gt[:, :14, ...])
        final_dice = self.loss_fn(final_pred, gt[:, :14])
        loss = overall_dice + final_weight * final_dice
        return loss.mean()


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

    optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    criterion = MyLossFunction()
    trainer = DualSegTrainer(args, net, train_loader, val_loader, criterion, optimizer, device="cuda:0")
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
