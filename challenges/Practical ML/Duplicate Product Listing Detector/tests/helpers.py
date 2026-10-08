"""Test code shared by several test modules (moved out of test files that used to import each other)."""

from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image


class FakeImage:
    dim = 16  # > number of groups, so no two groups share a vector

    def encode_images(self, images):
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        for i, im in enumerate(images):
            out[i, im.size[0] % self.dim] = 1.0
        return out


class FakeText:
    dim = 16  # > number of groups, so no two groups share a vector

    def __init__(self):
        self.calls = 0

    def encode_texts(self, texts):
        self.calls += 1
        # Same "product N" prefix -> same vector, so duplicates are text-identical.
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, int(t.split()[1]) % self.dim] = 1.0
        return out


def make_catalog(data_dir: Path, n_groups: int = 12) -> pl.DataFrame:
    img_dir = data_dir / "train_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    for g in range(n_groups):
        for j in range(2 + g % 2):
            name = f"{n}.jpg"
            Image.new("RGB", (10 + g, 10), (g * 10, 0, 0)).save(img_dir / name)
            rows.append(
                {
                    "posting_id": f"train_{n}",
                    "image": name,
                    "image_phash": f"{n:016x}",
                    "title": f"product {g} variant {j}",
                    "label_group": g,
                }
            )
            n += 1
    df = pl.DataFrame(rows)
    df.write_csv(data_dir / "train.csv")
    return df
