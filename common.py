"""Fungsi bersama: baca daftar kelas dan bangun model SSDLite320 + MobileNetV3-Large."""
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

ROOT = Path(__file__).resolve().parent


def load_classes(path=ROOT / "classes.txt"):
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def build_model(num_classes, mode="partial", unfreeze_blocks=4):
    """num_classes = jumlah kelas objek (tanpa background; ditambah otomatis).

    mode:
      feature : backbone beku, hanya head (klasifikasi + regresi) dilatih
      partial : beku sebagian besar backbone, buka `unfreeze_blocks` blok terakhir
                + blok akhir + extra layers + head
      scratch : tanpa bobot pretrained, semua dilatih dari nol
    """
    if mode not in ("feature", "partial", "scratch"):
        raise ValueError(f"mode tidak dikenal: {mode}")

    if mode == "scratch":
        model = ssdlite320_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=num_classes + 1
        )
        return model

    # bobot COCO (domain sumber) -> classifier lama dipotong, diganti yang baru
    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    )
    in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels, num_anchors, num_classes + 1, norm_layer
    )

    for p in model.backbone.parameters():
        p.requires_grad = False
    if mode == "partial":
        # features[0] = blok awal sampai C4, features[1] = blok akhir, extra = lapisan tambahan
        for blk in list(model.backbone.features[0])[-unfreeze_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in model.backbone.features[1].parameters():
            p.requires_grad = True
        for p in model.backbone.extra.parameters():
            p.requires_grad = True
    return model


def set_train_mode(model, mode):
    """Lapisan beku tetap eval() agar statistik BatchNorm COCO tidak berubah oleh batch kecil."""
    model.train()
    if mode == "scratch":
        return
    for m in model.backbone.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()


def param_groups(model, mode, lr_head=1e-3, lr_backbone=1e-4):
    """Discriminative LR (slide 12): backbone yang dibuka lebih kecil, head lebih besar."""
    head = [p for p in model.head.parameters() if p.requires_grad]
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    if mode == "scratch":
        return [{"params": head + backbone, "lr": lr_head}]
    groups = [{"params": head, "lr": lr_head}]
    if backbone:
        groups.append({"params": backbone, "lr": lr_backbone})
    return groups


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
