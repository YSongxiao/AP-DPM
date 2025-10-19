from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
import pandas as pd
import json
import cv2
import torch
from evaluations.metrics import overlap_pairs


class CarpalDataset(Dataset):
    def __init__(self, data_root, annotation_path, transform=None):
        coco = COCO(annotation_file=annotation_path)
        self.data_root = data_root
        self.transform = transform
        img_ids = coco.getImgIds()

        # 存储结果，每张图一个 mask 列表
        self.filenames = []
        self.masks = []

        for img_id in img_ids:
            img_info = coco.loadImgs([img_id])[0]
            height, width = img_info['height'], img_info['width']
            filename = img_info['file_name']
            # filename = img_info['file_name'].split("_bmp")[0].replace("-", "!") + ".bmp"

            # 获取 annotation
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)

            # 按 category_id 排序
            anns_sorted = sorted(anns, key=lambda x: x['category_id'])

            # 对每个 annotation 生成 mask
            image_masks = []
            for num, ann in enumerate(anns_sorted):
                seg = ann['segmentation']

                # Polygon 转 RLE
                if isinstance(seg, list):
                    rles = maskUtils.frPyObjects(seg, height, width)
                    rle = maskUtils.merge(rles)
                else:
                    rle = seg

                mask = maskUtils.decode(rle)  # H x W numpy 数组（0/1）
                image_masks.append(mask)
            self.filenames.append(filename)
            tmp_masks = np.stack(image_masks, axis=0)
            self.masks.append(tmp_masks)

    def __getitem__(self, idx):
        # 归一化
        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range

        path = Path(self.data_root) / self.filenames[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.filenames[idx][-4] == "L":
            img = cv2.flip(img, 1)  # Horizontal Flip
        mask = self.masks[idx].transpose(1, 2, 0)
        img = normalization(img[..., np.newaxis]).astype(np.float32)
        dic = self.transform(image=img, mask=mask)
        img = dic['image'].float()
        mask = dic['mask'].permute(2, 0, 1)
        data = {
            "fname": self.filenames[idx],
            "img": img,
            "mask": mask
        }
        return data

    def __len__(self):
        return len(self.filenames)


class CarpalNpyDataset(Dataset):
    def __init__(self, data_root, annotation_path, transform=None):
        self.data_root = data_root
        self.annotation_path = annotation_path
        self.transform = transform

        # 存储结果，每张图一个 mask 列表
        self.filenames = [str(fname.stem) for fname in Path(self.annotation_path).rglob("*.npy")]
        self.masks = []

        for filename in self.filenames:
            tmp_mask = np.load(Path(self.annotation_path) / (filename + ".npy"))
            self.masks.append(tmp_mask)

    def __getitem__(self, idx):
        # 归一化
        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range

        path = Path(self.data_root) / (self.filenames[idx] + ".bmp")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask = self.masks[idx]
        if self.filenames[idx][-1] == "L":
            img = cv2.flip(img, 1)  # Horizontal Flip
            mask = self.masks[idx][:, :, ::-1]
        mask = mask.transpose(1, 2, 0)
        img = normalization(img[..., np.newaxis]).astype(np.float32)
        dic = self.transform(image=img, mask=mask)
        img = dic['image'].float()
        mask = dic['mask'].permute(2, 0, 1)
        data = {
            "fname": self.filenames[idx],
            "img": img,
            "gt": mask
        }
        return data

    def __len__(self):
        return len(self.filenames)


class CarpalNpyDatasetWithOverlap(Dataset):
    def __init__(self, data_root, annotation_path, transform=None):
        self.data_root = data_root
        self.annotation_path = annotation_path
        self.transform = transform

        # 存储结果，每张图一个 mask 列表
        self.filenames = [str(fname.stem) for fname in Path(self.annotation_path).rglob("*.npy")]
        self.masks = []

        for filename in self.filenames:
            tmp_mask = np.load(Path(self.annotation_path) / (filename + ".npy"))
            self.masks.append(tmp_mask)

    def __getitem__(self, idx):
        # 归一化
        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range

        path = Path(self.data_root) / (self.filenames[idx] + ".bmp")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask = self.masks[idx]
        if self.filenames[idx][-1] == "L":
            img = cv2.flip(img, 1)  # Horizontal Flip
            mask = self.masks[idx][:, :, ::-1]
        mask = mask.transpose(1, 2, 0)
        img = normalization(img[..., np.newaxis]).astype(np.float32)
        dic = self.transform(image=img, mask=mask)
        img = dic['image'].float()
        mask = dic['mask'].permute(2, 0, 1)
        for (i, j) in overlap_pairs:
            overlap_mask = torch.logical_and(mask[i], mask[j])[None]
            mask = torch.concat([mask, overlap_mask], dim=0)
        data = {
            "fname": self.filenames[idx],
            "img": img,
            "gt": mask
        }
        return data

    def __len__(self):
        return len(self.filenames)


class CarpalNpyDatasetWithOverlapInfer(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform

        # 存储结果，每张图一个 mask 列表
        self.filenames = [str(fname.stem) for fname in Path(self.data_root).rglob("*.bmp")]

    def __getitem__(self, idx):
        # 归一化
        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range

        path = Path(self.data_root) / (self.filenames[idx] + ".bmp")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if self.filenames[idx][-1] == "L":
            img = cv2.flip(img, 1)  # Horizontal Flip

        img = normalization(img[..., np.newaxis]).astype(np.float32)
        dic = self.transform(image=img)
        img = dic['image'].float()

        data = {
            "fname": self.filenames[idx],
            "img": img,
        }
        return data

    def __len__(self):
        return len(self.filenames)


class CarpalNpyDatasetWithOverlapPlus(Dataset):
    def __init__(self, data_root, annotation_path, transform=None):
        self.data_root = data_root
        self.annotation_path = annotation_path
        self.transform = transform

        # 存储结果，每张图一个 mask 列表
        self.filenames = [str(fname.stem) for fname in Path(self.annotation_path).rglob("*.npy")]
        self.masks = []

        for filename in self.filenames:
            tmp_mask = np.load(Path(self.annotation_path) / (filename + ".npy"))
            self.masks.append(tmp_mask)

    def __getitem__(self, idx):
        # 归一化
        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range

        path = Path(self.data_root) / (self.filenames[idx] + ".bmp")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask = self.masks[idx]
        if self.filenames[idx][-1] == "L":
            img = cv2.flip(img, 1)  # Horizontal Flip
            mask = self.masks[idx][:, :, ::-1]
        mask = mask.transpose(1, 2, 0)
        img = normalization(img[..., np.newaxis]).astype(np.float32)
        dic = self.transform(image=img, mask=mask)
        img = dic['image'].float()
        mask = dic['mask'].permute(2, 0, 1)
        mask_overlap_semantic = mask.sum(dim=0, keepdim=True)
        mask_overlap_semantic = mask_overlap_semantic.float()
        # mask_overlap_semantic[mask_overlap_semantic > 1] = 1
        # mask_overlap_semantic[mask_overlap_semantic <= 1] = 0
        for (i, j) in overlap_pairs:
            overlap_mask = torch.logical_and(mask[i], mask[j])[None]
            mask = torch.concat([mask, overlap_mask], dim=0)
        data = {
            "fname": self.filenames[idx],
            "img": img,
            "gt": mask,
            "gt_sm": mask_overlap_semantic
        }
        return data

    def __len__(self):
        return len(self.filenames)


class CarpalNpyDatasetForPrior(Dataset):
    def __init__(self, data_root, annotation_path, transform=None):
        self.data_root = data_root
        self.annotation_path = annotation_path
        self.transform = transform

        # 存储结果，每张图一个 mask 列表
        self.filenames = [str(fname.stem) for fname in Path(self.annotation_path).rglob("*.npy")]
        self.masks = []

        for filename in self.filenames:
            tmp_mask = np.load(Path(self.annotation_path) / (filename + ".npy"))
            for i in range(tmp_mask.shape[0]):
                self.masks.append(tmp_mask[i][np.newaxis, ...])

    def __getitem__(self, idx):
        # 归一化
        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range

        img_idx = idx // 14
        path = Path(self.data_root) / (self.filenames[img_idx] + ".bmp")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask = self.masks[idx]
        if self.filenames[img_idx][-1] == "L":
            img = cv2.flip(img, 1)  # Horizontal Flip
            mask = self.masks[idx][:, :, ::-1]
        mask = mask.transpose(1, 2, 0)
        img = normalization(img[..., np.newaxis]).astype(np.float32)
        dic = self.transform(image=img, mask=mask)
        img = dic['image'].float()
        mask = dic['mask'].permute(2, 0, 1)
        data = {
            "fname": self.filenames[img_idx],
            "img": img,
            "gt": mask,
        }
        return data

    def __len__(self):
        return len(self.masks)


def get_dataloader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


if __name__ == '__main__':
    from utils import get_transform
    transform_val = get_transform(split="val", image_size=512)
    carpal_dataset = CarpalNpyDatasetWithOverlap(data_root="/mnt/data2/datasx/Carpal/ExportedDataset/FHM-W400_v1/Segmentation/image/", annotation_path="/mnt/data2/datasx/Carpal/ExportedDataset/FHM-W400_v1/Segmentation/mask/train/", transform=transform_val)
    data = carpal_dataset.__getitem__(0)
    import matplotlib.pyplot as plt
    plt.imsave("../figs/sample_img.png", data["img"].numpy()[0], cmap="gray")
    for i in range(14):
        plt.imsave(f"../figs/sample_bone_mask_{i}.png", data["gt"][i], cmap="gray")
        plt.imsave(f"../figs/sample_overlap_mask_{i}.png", data["gt"][i+14], cmap="gray")
