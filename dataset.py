"""Dataset deteksi format YOLO (txt: kelas cx cy w h, ternormalisasi) untuk torchvision."""
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


class YoloDetectionDataset(Dataset):
    def __init__(self, root, split, train=False, flip=True):
        root = Path(root)
        self.img_dir = root / "images" / split
        self.lbl_dir = root / "labels" / split
        self.files = sorted(p for p in self.img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        if train:
            ops = [
                v2.RandomPhotometricDistort(p=0.5),   # cahaya terang/redup
                v2.RandomZoomOut(fill=114, side_range=(1.0, 1.5), p=0.3),
                v2.RandomIoUCrop(),
                v2.Resize((320, 320)),
            ]
            if flip:
                ops.append(v2.RandomHorizontalFlip(0.5))
            ops += [v2.SanitizeBoundingBoxes(), v2.ToDtype(torch.float32, scale=True)]
            # RandomIoUCrop kadang membuang semua kotak; dipanggil ulang di __getitem__
            self.tf = v2.Compose(ops)
        else:
            self.tf = v2.Compose([v2.Resize((320, 320)), v2.ToDtype(torch.float32, scale=True)])
        self.train = train

    def __len__(self):
        return len(self.files)

    def _read(self, i):
        path = self.files[i]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # OpenCV = BGR, model = RGB
        h, w = img.shape[:2]
        boxes, labels = [], []
        lbl = self.lbl_dir / (path.stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(c + 1)                        # 0 = background
        img_t = tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1))
        boxes_t = tv_tensors.BoundingBoxes(
            torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            format="XYXY", canvas_size=(h, w),
        )
        return img_t, {"boxes": boxes_t, "labels": torch.tensor(labels, dtype=torch.int64)}

    def __getitem__(self, i):
        img, tgt = self._read(i)
        n_before = len(tgt["boxes"])
        for _ in range(10 if self.train else 1):
            out_img, out_tgt = self.tf(img, tgt)
            if len(out_tgt["boxes"]) > 0 or n_before == 0:
                break
        else:
            out_img, out_tgt = v2.Compose([v2.Resize((320, 320)),
                                           v2.ToDtype(torch.float32, scale=True)])(img, tgt)
        out_tgt = {"boxes": torch.as_tensor(out_tgt["boxes"]).float(),
                   "labels": out_tgt["labels"]}
        return out_img.as_subclass(torch.Tensor), out_tgt


def collate(batch):
    return tuple(zip(*batch))
