"""Test code shared by several test modules (moved out of test files that used to import each other)."""

import zlib

import numpy as np
import polars as pl


class HashEncoder:
    def encode(self, texts):
        out = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, zlib.crc32(w.encode()) % 64] += 1
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm == 0, 1, norm)


def _frames():
    rows_j, rows_r = [], []
    for cat, words in VOCAB.items():
        for i in range(6):
            body = " ".join((words * 8)[: 30 + i] + COMMON * 2)
            rows_j.append((f"{cat}-{i}", f"{cat.title()} role {i}", body, cat))
            rows_r.append(
                (f"r-{cat}-{i}", cat, " ".join((words * 6)[: 25 + i] + COMMON))
            )
    jobs = pl.DataFrame(
        rows_j, schema=["job_id", "title", "text", "category"], orient="row"
    )
    resumes = pl.DataFrame(rows_r, schema=["id", "category", "text"], orient="row")
    return jobs, resumes


COMMON = ["team", "worked", "years", "managed", "responsible", "experience"]


VOCAB = {
    "HEALTHCARE": ["nurse", "patient", "ward", "clinic", "dosage", "triage"],
    "CHEF": ["kitchen", "menu", "sauce", "saute", "pastry", "plating"],
    "TEACHER": ["classroom", "lesson", "pupils", "grading", "curriculum", "phonics"],
}
