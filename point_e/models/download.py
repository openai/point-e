"""
Adapted from: https://github.com/openai/glide-text2im/blob/69b530740eb6cef69442d6180579ef5ba9ef063e/glide_text2im/download.py
"""

import os
from functools import lru_cache
from typing import Dict, Optional

import requests
import torch
from filelock import FileLock
from tqdm.auto import tqdm

MODEL_PATHS = {
    "base40M-imagevec": "https://openaipublic.azureedge.net/main/point-e/base_40m_imagevec.pt",
    "base40M-textvec": "https://openaipublic.azureedge.net/main/point-e/base_40m_textvec.pt",
    "base40M-uncond": "https://openaipublic.azureedge.net/main/point-e/base_40m_uncond.pt",
    "base40M": "https://openaipublic.azureedge.net/main/point-e/base_40m.pt",
    "base300M": "https://openaipublic.azureedge.net/main/point-e/base_300m.pt",
    "base1B": "https://openaipublic.azureedge.net/main/point-e/base_1b.pt",
    "upsample": "https://openaipublic.azureedge.net/main/point-e/upsample_40m.pt",
    "sdf": "https://openaipublic.azureedge.net/main/point-e/sdf.pt",
    "pointnet": "https://openaipublic.azureedge.net/main/point-e/pointnet.pt",
}


@lru_cache()
def default_cache_dir() -> str:
    return os.path.join(os.path.abspath(os.getcwd()), "point_e_model_cache")


def fetch_file_cached(
    url: str, progress: bool = True, cache_dir: Optional[str] = None, chunk_size: int = 4096
) -> str:
    """
    Download the file at the given URL into a local file and return the path.
    If cache_dir is specified, it will be used to download the files.
    Otherwise, default_cache_dir() is used.
    """
    if cache_dir is None:
        cache_dir = default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(local_path):
        return local_path

    response = requests.get(url, stream=True)
    size = int(response.headers.get("content-length", "0"))
    with FileLock(local_path + ".lock"):
        if progress:
            pbar = tqdm(total=size, unit="iB", unit_scale=True)
        tmp_path = local_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if progress:
                    pbar.update(len(chunk))
                f.write(chunk)
        os.rename(tmp_path, local_path)
        if progress:
            pbar.close()
        return local_path


def load_checkpoint(
    checkpoint_name: str,
    device: torch.device,
    progress: bool = True,
    cache_dir: Optional[str] = None,
    chunk_size: int = 4096,
) -> Dict[str, torch.Tensor]:
    if checkpoint_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown checkpoint name {checkpoint_name}. Known names are: {MODEL_PATHS.keys()}."
        )
    path = fetch_file_cached(
        MODEL_PATHS[checkpoint_name], progress=progress, cache_dir=cache_dir, chunk_size=chunk_size
    )
    return torch.load(path, map_location=device)
