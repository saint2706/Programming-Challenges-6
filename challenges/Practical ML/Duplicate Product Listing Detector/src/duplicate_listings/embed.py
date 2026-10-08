"""Text and image embeddings for the catalog, cached to disk.

Two encoders, deliberately different models:

* ``MultilingualTextEncoder`` -- ``multilingual-e5-small`` for titles. Shopee
  titles mix Indonesian, Malay and English, so an English-only model would
  treat "kantong kertas" and "paper bag" as unrelated.
* ``SigLIP2ImageEncoder`` -- SigLIP 2's vision tower for product photos.

Both are hidden behind ``encode_texts`` / ``encode_images`` so the rest of the
pipeline (and every test except one smoke test) runs against tiny fakes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
from PIL import Image

TEXT_MODEL = "intfloat/multilingual-e5-small"
IMAGE_MODEL = "google/siglip2-base-patch16-224"


class TextEncoder(Protocol):
    dim: int

    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


class ImageEncoder(Protocol):
    dim: int

    def encode_images(self, images: list[Image.Image]) -> np.ndarray: ...


@dataclass
class Embeddings:
    posting_ids: list[str]
    text: np.ndarray  # (n, d_text), L2-normalized
    image: np.ndarray  # (n, d_image), L2-normalized; all-zero row = image unavailable
    missing_images: list[str]  # posting_ids whose image could not be read

    def subset(self, rows: list[int]) -> "Embeddings":
        """Rows in the given order, e.g. to carve a split out of the full slice."""
        ids = [self.posting_ids[i] for i in rows]
        missing = set(self.missing_images)
        return Embeddings(
            ids, self.text[rows], self.image[rows], [p for p in ids if p in missing]
        )


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise unit norm; all-zero rows stay zero (cosine 0) rather than NaN."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(
        x, norms, out=np.zeros_like(x, dtype=np.float32), where=norms > 0
    ).astype(np.float32)


class MultilingualTextEncoder:
    def __init__(self, model_name: str = TEXT_MODEL, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_embedding_dimension()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        # e5 models expect a task prefix; "query: " is the documented choice for
        # symmetric similarity (both sides are listings, not question/passage).
        return self.model.encode(
            [f"query: {t}" for t in texts], batch_size=64, show_progress_bar=False
        )


class SigLIP2ImageEncoder:
    def __init__(self, model_name: str = IMAGE_MODEL, device: str = "cpu"):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.dim = self.model.config.vision_config.hidden_size

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            feats = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        if not isinstance(
            feats, torch.Tensor
        ):  # newer transformers return a model-output object
            feats = feats.pooler_output
        return feats.float().cpu().numpy()


def _load_image(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except (OSError, ValueError):
        return None


def embed_catalog(
    df: pl.DataFrame,
    data_dir: Path,
    text_encoder: TextEncoder,
    image_encoder: ImageEncoder,
    batch_size: int = 32,
) -> Embeddings:
    """Embed every listing's title and image, preserving `df` row order.

    A missing or corrupt image never aborts the run: its row is all zeros (so it
    contributes no image similarity) and its posting_id is recorded.
    """
    ids = df["posting_id"].to_list()
    titles = df["title"].to_list()
    names = df["image"].to_list()
    n = len(ids)

    text = np.zeros((n, text_encoder.dim), dtype=np.float32)
    image = np.zeros((n, image_encoder.dim), dtype=np.float32)
    missing: list[str] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        text[start:end] = text_encoder.encode_texts(titles[start:end])

        loaded: list[tuple[int, Image.Image]] = []
        for i in range(start, end):
            im = _load_image(data_dir / "train_images" / names[i])
            if im is None:
                missing.append(ids[i])
            else:
                loaded.append((i, im))
        if loaded:
            vecs = image_encoder.encode_images([im for _, im in loaded])
            for (i, _), v in zip(loaded, vecs, strict=True):
                image[i] = v

    return Embeddings(ids, l2_normalize(text), l2_normalize(image), missing)


def save_embeddings(path: Path, emb: Embeddings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        posting_ids=np.array(emb.posting_ids),
        text=emb.text,
        image=emb.image,
        missing_images=np.array(emb.missing_images, dtype=str),
    )


def load_embeddings(path: Path, expected_ids: list[str]) -> Embeddings | None:
    """Return cached embeddings only if they cover exactly `expected_ids`, in order."""
    if not path.exists():
        return None
    with np.load(path) as z:
        ids = z["posting_ids"].tolist()
        if ids != expected_ids:
            return None
        return Embeddings(ids, z["text"], z["image"], z["missing_images"].tolist())
